"""Feature engineering pipeline to process movie, user, and interaction features.

This pipeline loads the merged dataset, applies feature engineering, and saves processed features.
It uses parallel processing and chunking for efficient handling of large datasets.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path to allow imports
project_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(project_dir))

from src.movie_recommender.features.movie_feature_engineering import engineer_movie_features
from src.movie_recommender.features.interaction_feature_engineering import engineer_interaction_features, create_enhanced_features
from src.movie_recommender.utils.logging import setup_logging
from src.movie_recommender.utils.parallel_processing import parallel_process, chunked_parallel_process, get_optimal_workers

# Setup logging
logger = logging.getLogger(__name__)
setup_logging()

def process_user_profile(user_data):
    """
    Process a single user's data to create their profile.
    
    Parameters:
    - user_data: Tuple of (user_id, user_ratings, movie_df)
    
    Returns:
    - Dictionary with user profile features or None if error
    """
    user_id, user_ratings, movie_df = user_data
    
    try:
        # Consider only movies with ratings >= 3.5 as liked movies
        liked_movies = user_ratings[user_ratings['rating'] >= 3.5]['movieId'].tolist()
        
        if not liked_movies:
            return None
            
        # Get movie data for liked movies
        liked_movie_data = movie_df[movie_df['movieId'].isin(liked_movies)]
        
        if liked_movie_data.empty:
            return None
        
        # Extract genre preferences
        all_genres = []
        for genre_list in liked_movie_data['genres'].str.split('|'):
            if isinstance(genre_list, list):
                all_genres.extend(genre_list)
        
        # Count genre frequencies and select top 5 genres
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(5).index.tolist() if not genre_counts.empty else []
        genre_preferences = '|'.join(top_genres)
        
        # Extract preferred directors (top 3)
        directors = liked_movie_data['crew'].dropna().tolist()
        dir_counts = pd.Series(directors).value_counts()
        preferred_directors = dir_counts.head(3).index.tolist() if not dir_counts.empty else []
        
        # Extract top actors (top 5)
        all_actors = []
        for cast_list in liked_movie_data['cast'].str.split('|'):
            if isinstance(cast_list, list):
                all_actors.extend(cast_list[:3])  # Consider only the top 3 actors per movie
        
        actor_counts = pd.Series(all_actors).value_counts()
        top_actors = actor_counts.head(5).index.tolist() if not actor_counts.empty else []
        
        # Extract preferred languages
        languages = liked_movie_data['original_language'].dropna().unique().tolist()
        
        # Calculate average year, budget, revenue
        if 'release_date' in liked_movie_data.columns and not liked_movie_data['release_date'].isna().all():
            avg_year = pd.to_datetime(liked_movie_data['release_date']).dt.year.mean()
        else:
            avg_year = np.nan
            
        avg_budget = liked_movie_data['budget'].mean() if 'budget' in liked_movie_data.columns else np.nan
        avg_revenue = liked_movie_data['revenue'].mean() if 'revenue' in liked_movie_data.columns else np.nan
        
        # Create user profile
        user_profile = {
            'userId': user_id,
            'genre_preferences': genre_preferences,
            'preferred_directors': preferred_directors,
            'top_actors': top_actors,
            'preferred_languages': languages,
            'avg_year': avg_year,
            'avg_budget': avg_budget,
            'avg_revenue': avg_revenue
        }
        
        return user_profile
        
    except Exception as e:
        logger.error(f"Error processing user {user_id}: {e}")
        return None


def create_user_profiles(ratings_df: pd.DataFrame, movie_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates user profiles with preferred genres, directors, actors and other preferences
    based on their rating history. Uses parallel processing for efficiency.
    
    Parameters:
    - ratings_df: DataFrame with ['userId', 'movieId', 'rating']
    - movie_df: DataFrame with movie features
    
    Returns:
    - DataFrame indexed by userId with user profile features
    """
    logger.info("Creating user profiles...")
    start_time = time.time()
    
    # Prepare data for parallel processing
    user_data = []
    for user_id, user_ratings in ratings_df.groupby('userId'):
        user_data.append((user_id, user_ratings, movie_df))
    
    # Process users in parallel
    n_workers = get_optimal_workers()
    logger.info(f"Processing {len(user_data)} users with {n_workers} workers")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_user = {executor.submit(process_user_profile, data): data[0] for data in user_data}
        
        # Collect results as they complete
        user_stats = []
        for future in tqdm(as_completed(future_to_user), total=len(user_data), desc="Processing users"):
            result = future.result()
            if result is not None:
                user_stats.append(result)
    
    # Create DataFrame from user profiles
    user_profiles_df = pd.DataFrame(user_stats)
    if not user_profiles_df.empty:
        user_profiles_df.set_index('userId', inplace=True)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Created {len(user_profiles_df)} user profiles in {elapsed_time:.2f} seconds")
    return user_profiles_df

def process_interaction_batch(batch_data):
    """
    Process a batch of interactions to create interaction features.
    
    Parameters:
    - batch_data: Tuple of (batch_df, user_profiles, movie_features)
    
    Returns:
    - DataFrame with interaction features
    """
    batch_df, user_profiles, movie_features = batch_data
    
    try:
        # Create interaction features for this batch
        batch_features = engineer_interaction_features(
            batch_df,
            user_profiles,
            movie_features
        )
        return batch_features
    except Exception as e:
        logger.error(f"Error processing interaction batch: {e}")
        return pd.DataFrame()


def main(input_filepath: str, output_dir: str, chunk_size: int = 50000, n_workers: int = None):
    """
    Main function to run the feature engineering pipeline.
    
    Args:
        input_filepath: Path to the input merged data file
        output_dir: Directory to save processed data
        chunk_size: Size of chunks to use when processing large files
    """
    logger.info(f"Starting feature engineering pipeline with input file: {input_filepath}")
    start_time = time.time()
    
    # Set number of workers for parallel processing
    if n_workers is None:
        n_workers = get_optimal_workers()
    logger.info(f"Using {n_workers} workers for parallel processing")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
      # Load data
    try:
        if input_filepath.endswith('.parquet'):
            # First, read file metadata to check schema and size
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(input_filepath)
            num_rows = parquet_file.metadata.num_rows
            logger.info(f"Parquet file contains {num_rows} rows")
            
            # Read only the first chunk to get schema and check required columns
            data_sample = parquet_file.read_row_group(0).to_pandas()
            
            # Process in chunks if file is large (more than 500,000 rows)
            use_chunks = num_rows > 500000
            
            if use_chunks:
                logger.info(f"File is large ({num_rows} rows). Will process in chunks.")
                # We'll read the data in chunks later when needed
                data = data_sample
            else:
                # For smaller files, read everything into memory
                data = pd.read_parquet(input_filepath)
                logger.info(f"Loaded data with shape: {data.shape}")
        elif input_filepath.endswith('.csv'):
            # For CSV, check file size first
            file_size = os.path.getsize(input_filepath) / (1024 * 1024)  # size in MB
            logger.info(f"CSV file size: {file_size:.2f} MB")
            
            # Process in chunks if file is large (more than 250 MB)
            use_chunks = file_size > 250
            
            if use_chunks:
                logger.info(f"File is large ({file_size:.2f} MB). Will process in chunks.")
                # Read only a sample first to get schema
                data_sample = pd.read_csv(input_filepath, nrows=1000)
                data = data_sample
            else:
                # For smaller files, read everything into memory
                data = pd.read_csv(input_filepath)
                logger.info(f"Loaded data with shape: {data.shape}")
        else:
            logger.error(f"Unsupported file format: {input_filepath}")
            return
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Check if required columns exist
    required_cols = ['userId', 'movieId', 'rating', 'genres', 'crew', 'cast', 
                    'original_language', 'release_date', 'budget', 'revenue']
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return
      # Process movie features
    logger.info("Engineering movie features...")
    try:
        # Get unique movies
        if 'use_chunks' in locals() and use_chunks and input_filepath.endswith('.parquet'):
            unique_movie_ids = set()
            movie_chunks = []
            parquet_file = pq.ParquetFile(input_filepath)
            
            for i in range(parquet_file.num_row_groups):
                chunk = parquet_file.read_row_group(i).to_pandas()
                # Process chunk
                chunk_unique = chunk.drop_duplicates('movieId')
                new_movies = chunk_unique[~chunk_unique['movieId'].isin(unique_movie_ids)]
                unique_movie_ids.update(new_movies['movieId'].tolist())
                movie_chunks.append(new_movies)
            
            movies_df = pd.concat(movie_chunks, ignore_index=True)
            logger.info(f"Processed {len(movies_df)} unique movies from row groups")
        else:
            # For smaller files, we already have all data in memory
            movies_df = data.drop_duplicates('movieId')
        
        # Engineer movie features
        logger.info(f"Engineering features for {len(movies_df)} unique movies")
        movie_features = engineer_movie_features(movies_df)
        
        # Set index for faster joins later
        if 'movieId' in movie_features.columns:
            movie_features.set_index('movieId', inplace=True)
        
        # Save movie features
        movie_features_path = os.path.join(output_dir, 'movie_features.parquet')
        movie_features.to_parquet(movie_features_path)
        logger.info(f"Saved movie features to {movie_features_path}")
    except Exception as e:
        logger.error(f"Error engineering movie features: {e}")
        return
      # Create user profiles
    try:
        if 'use_chunks' in locals() and use_chunks:
            # Get unique users from Parquet
            if input_filepath.endswith('.parquet'):
                parquet_file = pq.ParquetFile(input_filepath)
                unique_user_ids = set()
                for i in range(parquet_file.num_row_groups):
                    chunk = parquet_file.read_row_group(i).to_pandas()
                    unique_user_ids.update(chunk['userId'].unique())
            
            logger.info(f"Found {len(unique_user_ids)} unique users")
            
            # Process users in parallel batches
            batch_size = 2000
            user_batches = [list(unique_user_ids)[i:i+batch_size] for i in range(0, len(unique_user_ids), batch_size)]
            
            user_profiles_list = []
            
            with tqdm(total=len(user_batches), desc="Processing user batches") as pbar:
                for batch_idx, user_batch in enumerate(user_batches):
                    logger.info(f"Processing user batch {batch_idx+1}/{len(user_batches)}")
                    
                    # Get all ratings for users in this batch
                    batch_ratings = []
                    batch_movie_data = []
                    if input_filepath.endswith('.parquet'):
                        # Read using row groups
                        parquet_file = pq.ParquetFile(input_filepath)
                        for i in range(parquet_file.num_row_groups):
                            # Read entire row group
                            chunk = parquet_file.read_row_group(i).to_pandas()
                            # Filter for users in current batch
                            batch_chunk = chunk[chunk['userId'].isin(user_batch)]
                            if not batch_chunk.empty:
                                batch_ratings.append(batch_chunk[['userId', 'movieId', 'rating']])
                                batch_movie_data.append(batch_chunk)
                    else:  # CSV
                        # CSV can still use chunksize
                        for chunk in pd.read_csv(input_filepath, chunksize=chunk_size):
                            batch_chunk = chunk[chunk['userId'].isin(user_batch)]
                            if not batch_chunk.empty:
                                batch_ratings.append(batch_chunk[['userId', 'movieId', 'rating']])
                                batch_movie_data.append(batch_chunk)
            
                    if batch_ratings:
                        batch_ratings_df = pd.concat(batch_ratings)
                        batch_movie_df = pd.concat(batch_movie_data)
                        
                        # Create profiles for this batch
                        batch_profiles = create_user_profiles(batch_ratings_df, batch_movie_df)
                        user_profiles_list.append(batch_profiles)
                    
                    pbar.update(1)
                    
            # Combine all user profiles
            user_profiles = pd.concat(user_profiles_list) if user_profiles_list else pd.DataFrame()
        else:
            # For smaller files, process all ratings at once
            ratings_df = data[['userId', 'movieId', 'rating']].copy()
            user_profiles = create_user_profiles(ratings_df, data)
        
        # Save user profiles
        user_profiles_path = os.path.join(output_dir, 'user_profiles.parquet')
        user_profiles.to_parquet(user_profiles_path)
        logger.info(f"Saved user profiles to {user_profiles_path}")
    except Exception as e:
        logger.error(f"Error creating user profiles: {e}")
        return
      # Create interaction features
    try:
        # If we're processing chunks, we need to load all ratings
        if 'use_chunks' in locals() and use_chunks and input_filepath.endswith('.parquet'):
            ratings_chunks = []
            parquet_file = pq.ParquetFile(input_filepath)
            for i in range(parquet_file.num_row_groups):
                chunk = parquet_file.read_row_group(i).to_pandas()[['userId', 'movieId', 'rating']]
                ratings_chunks.append(chunk)
            ratings_df = pd.concat(ratings_chunks)
                    
            logger.info(f"Loaded {len(ratings_df)} ratings for interaction features")
        else:
            # For smaller files, we already have data in memory
            ratings_df = data[['userId', 'movieId', 'rating']].copy()
            
        # Process in batches if there are many ratings
        if len(ratings_df) > 1000000:
            logger.info("Processing interaction features in batches")
            
            # Split ratings into batches for parallel processing
            batch_size = 50000
            ratings_batches = [ratings_df.iloc[i:i+batch_size] for i in range(0, len(ratings_df), batch_size)]
            
            # Prepare batch data for parallel processing
            batch_data = [(batch_df, user_profiles, movie_features) for batch_df in ratings_batches]
            
            logger.info(f"Processing {len(batch_data)} interaction batches in parallel")
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_batch = {executor.submit(process_interaction_batch, data): i 
                                  for i, data in enumerate(batch_data)}
                
                # Collect results as they complete
                interaction_features_list = []
                for future in tqdm(as_completed(future_to_batch), total=len(batch_data), 
                                  desc="Processing interaction batches"):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            interaction_features_list.append(result)
                        logger.info(f"Completed interaction batch {batch_idx+1}/{len(batch_data)}")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
            
            # Combine all interaction features
            interaction_features = pd.concat(interaction_features_list) if interaction_features_list else pd.DataFrame()
        else:
            # For smaller datasets, process all at once
            interaction_features = engineer_interaction_features(
                ratings_df,
                user_profiles,
                movie_features
            )
        
        # Save interaction features
        interaction_path = os.path.join(output_dir, 'interaction_features.parquet')
        interaction_features.to_parquet(interaction_path)
        logger.info(f"Saved interaction features to {interaction_path}")
    except Exception as e:
        logger.error(f"Error engineering interaction features: {e}")
        return
    
    # Prepare final dataset for model training
    try:
        # If we're processing chunks, we need to load all ratings again
        if 'use_chunks' in locals() and use_chunks:
            # We already have ratings_df from interaction features step
            pass
        else:
            # For smaller files, we already have data in memory
            ratings_df = data[['userId', 'movieId', 'rating']].copy()
        
        # Get original ratings
        final_df = ratings_df.copy()
        
        # Rename interaction feature columns to match
        interaction_features.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}, inplace=True)
        
        # Process in batches to avoid memory issues
        if len(final_df) > 1000000:
            logger.info("Processing final dataset in batches")
            
            # Split into batches
            batch_size = 100000
            df_batches = [final_df.iloc[i:i+batch_size] for i in range(0, len(final_df), batch_size)]
            
            final_batches = []
            
            for batch_idx, batch_df in enumerate(df_batches):
                logger.info(f"Processing final dataset batch {batch_idx+1}/{len(df_batches)}")
                
                # Add movie features
                for col in movie_features.columns:
                    if col not in ['movieId']:  # Skip index
                        batch_df[f'movie_{col}'] = batch_df['movieId'].map(movie_features[col])
                
                # Add user profile features
                for col in user_profiles.columns:
                    if col not in ['userId']:  # Skip index
                        batch_df[f'user_{col}'] = batch_df['userId'].map(user_profiles[col])
                
                # Get interaction features for this batch
                interaction_cols = [col for col in interaction_features.columns 
                                 if col not in ['userId', 'movieId']]
                
                # Merge interaction features
                batch_with_interactions = pd.merge(
                    batch_df, 
                    interaction_features[['userId', 'movieId'] + interaction_cols],
                    on=['userId', 'movieId'],
                    how='left'
                )
                
                final_batches.append(batch_with_interactions)
            
            # Combine all batches
            final_df = pd.concat(final_batches)
        else:
            # For smaller datasets, add features all at once
            # Add movie features
            for col in movie_features.columns:
                if col not in ['movieId']:  # Skip index
                    final_df[f'movie_{col}'] = final_df['movieId'].map(movie_features[col])
            
            # Add user profile features
            for col in user_profiles.columns:
                if col not in ['userId']:  # Skip index
                    final_df[f'user_{col}'] = final_df['userId'].map(user_profiles[col])
            
            # Get interaction features
            interaction_cols = [col for col in interaction_features.columns 
                             if col not in ['userId', 'movieId']]
            
            # Merge interaction features
            final_df = pd.merge(
                final_df, 
                interaction_features[['userId', 'movieId'] + interaction_cols],
                on=['userId', 'movieId'],
                how='left'
            )
          # Save final dataset
        final_path = os.path.join(output_dir, 'final_features.parquet')
        
        # For very large datasets, save in chunks to avoid memory issues
        if len(final_df) > 2000000:
            logger.info(f"Saving final dataset in chunks due to large size: {len(final_df)} rows")
            chunk_size = 500000
            for i in range(0, len(final_df), chunk_size):
                chunk_df = final_df.iloc[i:i+chunk_size]
                chunk_path = os.path.join(output_dir, f'final_features_chunk_{i//chunk_size}.parquet')
                chunk_df.to_parquet(chunk_path)
                logger.info(f"Saved chunk {i//chunk_size} with {len(chunk_df)} rows to {chunk_path}")
                
            # Create a small metadata file to track all chunks
            meta = pd.DataFrame({
                'chunk_path': [f'final_features_chunk_{i//chunk_size}.parquet' for i in range(0, len(final_df), chunk_size)],
                'chunk_size': [min(chunk_size, len(final_df) - i) for i in range(0, len(final_df), chunk_size)],
                'chunk_index': [i//chunk_size for i in range(0, len(final_df), chunk_size)]
            })
            meta.to_parquet(os.path.join(output_dir, 'final_features_chunks_meta.parquet'))
            logger.info(f"Saved chunks metadata to {os.path.join(output_dir, 'final_features_chunks_meta.parquet')}")
        else:
            # For smaller datasets, save in one file
            final_df.to_parquet(final_path)
            logger.info(f"Saved final dataset with {len(final_df)} rows to {final_path}")
    except Exception as e:
        logger.error(f"Error preparing final dataset: {e}")
        return
    
    logger.info("Feature engineering pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature engineering pipeline for movie recommendation')
    parser.add_argument('--input_filepath', type=str, 
                      default='data/interim/final_merged_data.parquet',
                      help='Path to the input merged data file')
    parser.add_argument('--output_dir', type=str, 
                     default='data/processed',
                     help='Directory to save processed features')
    parser.add_argument('--chunk_size', type=int, 
                     default=50000,
                     help='Size of chunks to use when processing large files')
    parser.add_argument('--workers', type=int, 
                     default=None,
                     help='Number of worker processes for parallel processing')
    
    args = parser.parse_args()
    
    # Print execution summary at the end
    start_time = time.time()
    main(args.input_filepath, args.output_dir, args.chunk_size, args.workers)
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"\nFeature engineering completed in {elapsed_time:.2f} seconds")
    print(f"Processed features saved to: {args.output_dir}")