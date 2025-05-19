# Prediction script for the neuro-fuzzy recommendation model

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
MODELS_DIR = PROJECT_DIR / 'artifacts' / 'models'

def load_model_and_mappings():
    """Load the trained model and mappings."""
    try:
        # Load model
        model_path = MODELS_DIR / 'final_model.h5'
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return None, None
        
        model = load_model(str(model_path))
        
        # Load mappings
        mappings_path = MODELS_DIR / 'mappings.joblib'
        if not mappings_path.exists():
            logger.error(f"Mappings file not found at {mappings_path}")
            return model, None
        
        mappings = joblib.load(mappings_path)
        
        logger.info("Successfully loaded model and mappings")
        return model, mappings
        
    except Exception as e:
        logger.error(f"Error loading model and mappings: {e}")
        return None, None

def load_user_item_matrix():
    """Load the user-item matrix."""
    try:
        # Load user-item matrix
        user_item_matrix_path = PROCESSED_DATA_DIR / 'user_item_matrix.csv'
        if not user_item_matrix_path.exists():
            logger.error(f"User-item matrix not found at {user_item_matrix_path}")
            return None
        
        user_item_df = pd.read_csv(user_item_matrix_path, index_col=0)
        
        logger.info(f"Loaded user-item matrix with shape {user_item_df.shape}")
        return user_item_df
        
    except Exception as e:
        logger.error(f"Error loading user-item matrix: {e}")
        return None

def get_recommendations_for_user(user_id, count=10):
    """Get movie recommendations for a specific user.
    
    Args:
        user_id (int): The ID of the user to get recommendations for.
        count (int): The number of recommendations to return.
        
    Returns:
        list: A list of recommended movie IDs.
    """
    try:
        # Load model and mappings
        model, mappings = load_model_and_mappings()
        if model is None or mappings is None:
            logger.error("Cannot generate recommendations: missing model or mappings")
            return []
        
        # Load user-item matrix
        user_item_df = load_user_item_matrix()
        if user_item_df is None:
            logger.error("Cannot generate recommendations: missing user-item matrix")
            return []
        
        # Check if user exists in the mappings
        user_mapping = mappings['user_mapping']
        item_mapping = mappings['item_mapping']
        
        # Find the internal user index
        user_idx = None
        for idx, uid in user_mapping.items():
            if uid == user_id:
                user_idx = idx
                break
        
        if user_idx is None:
            logger.warning(f"User ID {user_id} not found in mappings")
            # Return some popular movies as fallback
            return get_popular_movies(user_item_df, count)
        
        # Get all items
        n_items = mappings['n_items']
        item_indices = np.arange(n_items)
        
        # Create user-item pairs for prediction
        user_array = np.full(len(item_indices), user_idx)
        
        # Make predictions
        predictions = model.predict([user_array, item_indices], verbose=0)
        predictions = predictions.flatten()
        
        # Get the indices of the top N predictions
        # Exclude items the user has already rated
        user_row = user_item_df.iloc[user_idx]
        already_rated = user_row[user_row > 0].index.tolist()
        already_rated_indices = [int(i) for i in range(len(item_indices)) 
                               if item_mapping[i] in already_rated]
        
        # Set predictions for already rated items to -1
        predictions[already_rated_indices] = -1
        
        # Get top N recommendations
        top_n_indices = np.argsort(predictions)[-count:][::-1]
        
        # Convert internal indices to movie IDs
        recommended_movie_ids = [item_mapping[idx] for idx in top_n_indices]
        
        logger.info(f"Generated {len(recommended_movie_ids)} recommendations for user {user_id}")
        return recommended_movie_ids
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

def get_popular_movies(user_item_df, count=10):
    """Get popular movies as a fallback recommendation strategy.
    
    Args:
        user_item_df (DataFrame): The user-item matrix.
        count (int): The number of recommendations to return.
        
    Returns:
        list: A list of popular movie IDs.
    """
    try:
        # Calculate the sum of ratings for each movie
        movie_popularity = user_item_df.sum().sort_values(ascending=False)
        
        # Get the top N movie IDs
        popular_movie_ids = movie_popularity.index[:count].tolist()
        popular_movie_ids = [int(mid) for mid in popular_movie_ids]
        
        return popular_movie_ids
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        return []

# For testing
if __name__ == "__main__":
    # Test the recommendation function
    user_id = 1  # Example user ID
    recommendations = get_recommendations_for_user(user_id, 5)
    print(f"Recommendations for user {user_id}: {recommendations}")