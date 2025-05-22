import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def process_interaction_row(row):
    """
    Process a single interaction row to compute interaction features.
    
    Parameters:
    - row: Series containing merged user and movie data
    
    Returns:
    - Dictionary with interaction features
    """
    try:
        features = {}
        
        # Genre affinity using Jaccard similarity
        user_genres = set(str(row.get('genre_preferences', '')).split('|'))
        movie_genres = set(str(row.get('genres', '')).split('|'))
        # Remove empty strings
        user_genres.discard('')
        movie_genres.discard('')
        
        intersection = len(user_genres & movie_genres)
        union = len(user_genres | movie_genres)
        features['genre_affinity'] = intersection / union if union > 0 else 0.0

        # Director match
        movie_director = str(row.get('crew', ''))
        user_directors = row.get('preferred_directors', [])
        # Convert to list if it's a string
        if isinstance(user_directors, str):
            user_directors = eval(user_directors)
        features['director_match'] = 1 if movie_director in user_directors else 0

        # Actor overlap
        movie_actors = set(str(row.get('cast', '')).split('|'))
        user_actors = set(row.get('top_actors', []))
        # Convert to list if it's a string
        if isinstance(user_actors, str):
            user_actors = eval(user_actors)
        # Remove empty strings
        movie_actors.discard('')
        
        if movie_actors:
            matches = len(movie_actors & user_actors)
            features['actor_overlap'] = matches / len(movie_actors)
        else:
            features['actor_overlap'] = 0.0

        # Language match
        user_languages = row.get('preferred_languages', [])
        # Convert to list if it's a string
        if isinstance(user_languages, str):
            user_languages = eval(user_languages)
            
        features['language_match'] = 1 if (
            row.get('original_language') in user_languages
        ) else 0

        # Year distance
        if pd.notna(row.get('release_date')) and pd.notna(row.get('avg_year')):
            try:
                year = pd.to_datetime(row['release_date']).year
                features['year_distance'] = abs(year - row['avg_year'])
            except:
                features['year_distance'] = np.nan
        else:
            features['year_distance'] = np.nan

        # Budget distance
        if pd.notna(row.get('budget')) and pd.notna(row.get('avg_budget')):
            features['budget_distance'] = abs(row['budget'] - row['avg_budget'])
        else:
            features['budget_distance'] = np.nan

        # Revenue distance
        if pd.notna(row.get('revenue')) and pd.notna(row.get('avg_revenue')):
            features['revenue_distance'] = abs(row['revenue'] - row['avg_revenue'])
        else:
            features['revenue_distance'] = np.nan

        # Add user and movie IDs
        features['user_id'] = row['userId']
        features['movie_id'] = row['movieId']
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        return None


def engineer_interaction_features(
    ratings: pd.DataFrame,
    user_profiles: pd.DataFrame,
    movie_features: pd.DataFrame,
    genre_cols: List[str] = None,
    batch_size: int = 10000,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Compute user-item interaction features for each rating record.

    Parameters:
    - ratings: DataFrame with columns ['userId', 'movieId', 'rating']
    - user_profiles: DataFrame indexed by userId containing:
        * genre_preferences: pipe-separated string of preferred genres
        * preferred_directors: list of director names
        * top_actors: list of actor names
        * preferred_languages: list of language codes
        * avg_year: numeric
        * avg_budget: numeric
        * avg_revenue: numeric
    - movie_features: DataFrame indexed by movieId containing:
        * genres: pipe-separated string of genres
        * crew: director name
        * cast: pipe-separated string of actor names
        * original_language: language code
        * release_date: datetime
        * budget: numeric
        * revenue: numeric
    - genre_cols: optional list of genre column names (if using one-hot encoding)
    - batch_size: size of batches for processing
    - n_workers: number of worker processes for parallel processing

    Returns:
    - DataFrame with interaction feature columns
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    start_time = time.time()
    logger.info("Starting interaction feature engineering")
    
    # Determine optimal number of workers if not specified
    if n_workers is None:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Make a copy to avoid modifying the original
    df = ratings.copy()
    
    # Reset index on user_profiles if it's not already
    if user_profiles.index.name == 'userId':
        user_profiles = user_profiles.reset_index()
    
    # Reset index on movie_features if it's not already
    if movie_features.index.name == 'movieId':
        movie_features = movie_features.reset_index()
    
    # Merge user profiles and movie features
    logger.info("Merging user profiles and movie features")
    df = pd.merge(df, user_profiles, on='userId', how='left')
    df = pd.merge(df, movie_features, on='movieId', how='left')
    
    logger.info(f"Merged data shape: {df.shape}")
    
    # Process in batches to avoid memory issues
    features_list = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(df)} interactions in {total_batches} batches with {n_workers} workers")
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_df)} rows")
        
        # Process rows in parallel
        batch_features = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Convert DataFrame rows to dictionaries for easier processing
            row_dicts = batch_df.to_dict('records')
            
            # Submit all tasks
            future_to_row = {executor.submit(process_interaction_row, row): i 
                            for i, row in enumerate(row_dicts)}
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_row), total=len(row_dicts), 
                              desc=f"Batch {batch_idx+1}/{total_batches}"):
                result = future.result()
                if result is not None:
                    batch_features.append(result)
        
        features_list.extend(batch_features)
        logger.info(f"Completed batch {batch_idx+1}/{total_batches} with {len(batch_features)} valid features")
    
    # Convert to DataFrame
    if not features_list:
        logger.warning("No valid interaction features were generated")
        return pd.DataFrame()
    
    interaction_df = pd.DataFrame(features_list)
    
    # Create enhanced features
    interaction_df = create_enhanced_features(interaction_df)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed interaction feature engineering with {len(interaction_df)} records in {elapsed_time:.2f} seconds")
    return interaction_df

def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced interaction features by combining existing ones.
    
    Parameters:
    - df: DataFrame with basic interaction features
    
    Returns:
    - DataFrame with additional enhanced features
    """
    df = df.copy()
    
    # Content match score (genre and actor based)
    if all(col in df.columns for col in ['genre_affinity', 'actor_overlap']):
        df['content_match_score'] = (
            0.6 * df['genre_affinity'] + 
            0.4 * df['actor_overlap']
        )
    
    # Metadata match score (director and language based)
    if all(col in df.columns for col in ['director_match', 'language_match']):
        df['metadata_match_score'] = (
            0.7 * df['director_match'] + 
            0.3 * df['language_match']
        )
    
    # Production value match (budget and revenue based)
    if all(col in df.columns for col in ['budget_distance', 'revenue_distance']):
        df['production_value_match'] = (
            df['budget_distance'] * df['revenue_distance']
        )**0.5  # Geometric mean
    
    # Create interaction terms
    if all(col in df.columns for col in ['genre_affinity', 'director_match']):
        df['genre_director_synergy'] = df['genre_affinity'] * df['director_match']
    
    if all(col in df.columns for col in ['actor_overlap', 'language_match']):
        df['actor_language_synergy'] = df['actor_overlap'] * df['language_match']
    
    return df

# Example usage:
# interaction_df = engineer_interaction_features(
#     ratings_df,
#     user_profiles_df,
#     movie_features_df
# )
