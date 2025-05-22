# Data preparation script

import os
import logging
import yaml
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
CONFIG_PATH = PROJECT_DIR / 'configs' / 'model_params.yaml'

def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data():
    """Load raw data from the raw data directory."""
    logger.info(f"Loading raw data from {RAW_DATA_DIR}")
    
    try:
        # Load all required data files
        movies_path = RAW_DATA_DIR / 'movies.csv'
        ratings_path = RAW_DATA_DIR / 'ratings.csv'
        links_path = RAW_DATA_DIR / 'links.csv'
        tags_path = RAW_DATA_DIR / 'tags.csv'
        
        # Check if all required files exist
        required_files = [movies_path, ratings_path]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            logger.warning(f"Missing required data files: {missing_files}")
            return None, None
            
        # Load required datasets
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        
        # Load optional datasets if available
        try:
            if links_path.exists():
                links_df = pd.read_csv(links_path)
                movies_df = pd.merge(movies_df, links_df, on='movieId', how='left')
        except Exception as e:
            logger.warning(f"Error loading links data: {e}")
        
        try:
            if tags_path.exists():
                tags_df = pd.read_csv(tags_path)
                # Aggregate tags for each movie
                movie_tags = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
                movies_df = pd.merge(movies_df, movie_tags, on='movieId', how='left')
        except Exception as e:
            logger.warning(f"Error loading tags data: {e}")
        
        logger.info(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        return movies_df, ratings_df
        
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        return None, None

def preprocess_data(movies_df, ratings_df, config):
    """Preprocess the raw data for modeling."""
    if movies_df is None or ratings_df is None:
        logger.error("Cannot preprocess: missing input data")
        return None, None
    
    logger.info("Preprocessing data...")
    
    # Example preprocessing steps (adjust based on your needs):
    # 1. Clean movie titles and extract year
    # 2. Parse genres into a usable format
    # 3. Filter out users/movies with too few ratings
    # 4. Create user-item interaction matrix
    
    try:
        # Extract year from title and clean title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Convert genres from pipe-separated string to list
        movies_df['genres'] = movies_df['genres'].str.split('|')
        
        # Filter ratings based on minimum counts
        min_user_ratings = 5  # Minimum ratings per user
        min_movie_ratings = 10  # Minimum ratings per movie
        
        # Count ratings per user and movie
        user_counts = ratings_df['userId'].value_counts()
        movie_counts = ratings_df['movieId'].value_counts()
        
        # Filter users and movies with enough ratings
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        
        # Filter ratings
        filtered_ratings = ratings_df[
            ratings_df['userId'].isin(valid_users) & 
            ratings_df['movieId'].isin(valid_movies)
        ]
        
        # Create user-item matrix
        user_item_matrix = filtered_ratings.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Filter movies to only include those in the filtered ratings
        filtered_movies = movies_df[movies_df['movieId'].isin(valid_movies)]
        
        logger.info(f"Preprocessing complete. Retained {len(filtered_ratings)} ratings, "
                   f"{user_item_matrix.shape[0]} users, and {len(filtered_movies)} movies.")
        
        return filtered_movies, user_item_matrix
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None, None

def save_processed_data(movies_df, user_item_matrix):
    """Save processed data to the processed data directory."""
    if movies_df is None or user_item_matrix is None:
        logger.error("Cannot save: missing processed data")
        return False
    
    logger.info(f"Saving processed data to {PROCESSED_DATA_DIR}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Save processed data
        movies_df.to_csv(PROCESSED_DATA_DIR / 'processed_movies.csv', index=False)
        user_item_matrix.to_csv(PROCESSED_DATA_DIR / 'user_item_matrix.csv')
        
        # Save additional metadata for later use
        metadata = {
            'n_users': user_item_matrix.shape[0],
            'n_movies': user_item_matrix.shape[1],
            'n_ratings': user_item_matrix.values.nonzero()[0].size,
            'sparsity': user_item_matrix.values.nonzero()[0].size / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        }
        
        pd.DataFrame([metadata]).to_csv(PROCESSED_DATA_DIR / 'metadata.csv', index=False)
        
        logger.info("Processed data saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return False

def main():
    """Main function to run the data preparation pipeline."""
    logger.info("Starting data preparation pipeline")
    
    # Load configuration
    config = load_config()
    
    # Load raw data
    movies_df, ratings_df = load_raw_data()
    
    # Preprocess data
    processed_movies, user_item_matrix = preprocess_data(movies_df, ratings_df, config)
    
    # Save processed data
    success = save_processed_data(processed_movies, user_item_matrix)
    
    if success:
        logger.info("Data preparation pipeline completed successfully")
    else:
        logger.error("Data preparation pipeline failed")

if __name__ == "__main__":
    main()