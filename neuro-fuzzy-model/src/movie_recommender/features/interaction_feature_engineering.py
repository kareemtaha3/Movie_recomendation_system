import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def engineer_interaction_features(
    ratings: pd.DataFrame,
    user_profiles: pd.DataFrame,
    movie_features: pd.DataFrame,
    genre_cols: List[str] = None
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

    Returns:
    - DataFrame with interaction feature columns
    """
    logger.info("Starting interaction feature engineering")
    df = ratings.copy()

    # Merge in user_profile features
    df = df.join(user_profiles, on='userId', rsuffix='_user')
    logger.info(f"Merged user profiles for {len(df)} ratings")

    # Merge in movie_features
    df = df.join(movie_features, on='movieId', rsuffix='_movie')
    logger.info(f"Merged movie features for {len(df)} ratings")

    # Calculate interaction features
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing interaction features"):
        try:
            features = {}
            
            # Genre affinity using Jaccard similarity
            user_genres = set(row.get('genre_preferences', '').split('|'))
            movie_genres = set(row.get('genres', '').split('|'))
            intersection = len(user_genres & movie_genres)
            union = len(user_genres | movie_genres)
            features['genre_affinity'] = intersection / union if union > 0 else 0.0

            # Director match
            movie_director = row.get('crew', '')
            user_directors = row.get('preferred_directors', [])
            features['director_match'] = 1 if movie_director in user_directors else 0

            # Actor overlap
            movie_actors = set(row.get('cast', '').split('|'))
            user_actors = set(row.get('top_actors', []))
            if movie_actors:
                matches = len(movie_actors & user_actors)
                features['actor_overlap'] = matches / len(movie_actors)
            else:
                features['actor_overlap'] = 0.0

            # Language match
            features['language_match'] = 1 if (
                row.get('original_language') in row.get('preferred_languages', [])
            ) else 0

            # Year distance
            if pd.notna(row.get('release_date')) and pd.notna(row.get('avg_year')):
                year = pd.to_datetime(row['release_date']).year
                features['year_distance'] = abs(year - row['avg_year'])
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
            
            features_list.append(features)
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    # Convert to DataFrame
    interaction_df = pd.DataFrame(features_list)
    
    # Create enhanced features
    interaction_df = create_enhanced_features(interaction_df)
    
    logger.info(f"Completed interaction feature engineering with {len(interaction_df)} records")
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
