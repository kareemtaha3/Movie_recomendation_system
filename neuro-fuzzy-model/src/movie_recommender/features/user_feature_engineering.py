import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def normalize_numeric_features(df, columns=None, method='standard'):
    """
    Normalize numeric features using either standard scaling or min-max scaling.
    
    Args:
        df: DataFrame containing the features
        columns: List of columns to normalize (if None, all numeric columns are used)
        method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        
    Returns:
        DataFrame with normalized features and the scaler object
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    df_copy = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    else:  # minmax
        scaler = MinMaxScaler()
    
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy, scaler

def create_genre_embeddings(df, genre_columns):
    """
    Create genre embeddings using PCA
    
    Args:
        df: DataFrame containing genre proportion columns
        genre_columns: List of genre proportion columns (e.g., 'user_genre_proportion_ACTION')
        
    Returns:
        DataFrame with genre embedding features and the PCA object
    """
    df_copy = df.copy()
    pca = PCA(n_components=min(3, len(genre_columns)))
    
    genre_embeddings = pca.fit_transform(df[genre_columns])
    
    for i in range(genre_embeddings.shape[1]):
        df_copy[f'genre_embedding_{i+1}'] = genre_embeddings[:, i]
    
    return df_copy, pca

def create_all_features(df, genre_columns=None):
    """
    Apply all feature engineering functions
    
    Args:
        df: DataFrame containing the raw features
        genre_columns: List of genre proportion columns
        
    Returns:
        DataFrame with all engineered features
    """
    # If genre_columns is not provided, try to identify them from column names
    if genre_columns is None:
        genre_columns = [col for col in df.columns if col.startswith('user_genre_proportion_')]
        
    if genre_columns and len(genre_columns) > 1:
        df, _ = create_genre_embeddings(df, genre_columns)
    
    # Finally normalize everything
    df, _ = normalize_numeric_features(df)
    
    return df
        

def calculate_genre_affinity(user_genre_prefs, movie_genres):
    """
    Calculate the dot product between user genre preferences and movie genres
    
    Args:
        user_genre_prefs: Vector or dictionary of user genre preferences
        movie_genres: Vector or dictionary of movie genres (binary or weighted)
        
    Returns:
        Float representing genre affinity score
    """
    # If inputs are dictionaries, convert to aligned vectors
    if isinstance(user_genre_prefs, dict) and isinstance(movie_genres, dict):
        all_genres = set(user_genre_prefs.keys()).union(movie_genres.keys())
        user_vec = np.array([user_genre_prefs.get(g, 0) for g in all_genres])
        movie_vec = np.array([movie_genres.get(g, 0) for g in all_genres])
        return np.dot(user_vec, movie_vec)
    else:
        # Assume aligned vectors
        return np.dot(user_genre_prefs, movie_genres)

def get_director_match(movie_director, user_preferred_directors):
    """
    Check if the movie's director is in the user's preferred directors list
    
    Args:
        movie_director: String or ID of the movie's director
        user_preferred_directors: List or set of user's preferred directors
        
    Returns:
        1 if there's a match, 0 otherwise
    """
    return 1 if movie_director in user_preferred_directors else 0

def calculate_actor_overlap(movie_actors, user_top_n_actors):
    """
    Calculate the fraction of movie actors that are in the user's top-N preferred actors
    
    Args:
        movie_actors: List of actors in the movie
        user_top_n_actors: List or set of user's top-N preferred actors
        
    Returns:
        Fraction of overlap (0 to 1)
    """
    if not movie_actors:
        return 0
    
    user_actors_set = set(user_top_n_actors)
    matches = sum(1 for actor in movie_actors if actor in user_actors_set)
    return matches / len(movie_actors)

def calculate_keyword_similarity(user_keyword_vector, movie_keyword_vector):
    """
    Calculate cosine similarity between user and movie keyword vectors
    
    Args:
        user_keyword_vector: Vector representing user keyword preferences
        movie_keyword_vector: Vector representing movie keywords
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Reshape vectors if needed
    user_vec = user_keyword_vector.reshape(1, -1) if len(user_keyword_vector.shape) == 1 else user_keyword_vector
    movie_vec = movie_keyword_vector.reshape(1, -1) if len(movie_keyword_vector.shape) == 1 else movie_keyword_vector
    
    return cosine_similarity(user_vec, movie_vec)[0][0]

def generate_interaction_features(user_data, movie_data):
    """
    Generate all interaction features between a user and a movie
    
    Args:
        user_data: Dictionary or DataFrame row containing user features
        movie_data: Dictionary or DataFrame row containing movie features
        
    Returns:
        Dictionary of interaction features
    """
    features = {}
    
    # Genre affinity
    features['genre_affinity'] = calculate_genre_affinity(
        user_data['genre_preferences'], 
        movie_data['genres']
    )
    
    # Director match
    features['director_match'] = get_director_match(
        movie_data['director'], 
        user_data['preferred_directors']
    )
    
    # Actor overlap
    features['actor_overlap'] = calculate_actor_overlap(
        movie_data['actors'], 
        user_data['top_actors']
    )
    
    # Keyword similarity
    features['keyword_similarity'] = calculate_keyword_similarity(
        user_data['keyword_vector'], 
        movie_data['keyword_vector']
    )
    
    return features

def batch_interaction_features(users_df, movies_df, user_movie_pairs):
    """
    Generate interaction features for a batch of user-movie pairs
    
    Args:
        users_df: DataFrame containing user features
        movies_df: DataFrame containing movie features
        user_movie_pairs: List of (user_id, movie_id) tuples
        
    Returns:
        DataFrame of interaction features for each user-movie pair
    """
    features_list = []
    
    for user_id, movie_id in user_movie_pairs:
        user_data = users_df.loc[user_id].to_dict() if user_id in users_df.index else {}
        movie_data = movies_df.loc[movie_id].to_dict() if movie_id in movies_df.index else {}
        
        if user_data and movie_data:
            features = generate_interaction_features(user_data, movie_data)
            features['user_id'] = user_id
            features['movie_id'] = movie_id
            features_list.append(features)
    
    return pd.DataFrame(features_list)

def create_enhanced_interaction_features(interaction_features_df):
    """
    Create enhanced interaction features by combining existing ones
    
    Args:
        interaction_features_df: DataFrame containing basic interaction features
        
    Returns:
        DataFrame with additional enhanced features
    """
    df = interaction_features_df.copy()
    
    # Overall content match score (weighted combination of content-based factors)
    if all(col in df.columns for col in ['genre_affinity', 'keyword_similarity', 'actor_overlap']):
        df['content_match_score'] = (
            0.5 * df['genre_affinity'] + 
            0.3 * df['keyword_similarity'] + 
            0.2 * df['actor_overlap']
        )
    
    # Metadata match score (weighted combination of metadata factors)
    if all(col in df.columns for col in ['director_match', 'language_match']):
        df['metadata_match_score'] = (
            0.7 * df['director_match'] + 
            0.3 * df['language_match']
        )
    
    # Production value match (combination of budget and revenue match)
    if all(col in df.columns for col in ['budget_distance', 'revenue_distance']):
        df['production_value_match'] = (
            df['budget_distance'] * df['revenue_distance']
        )**0.5  # Geometric mean
    
    # Create interaction terms
    if all(col in df.columns for col in ['genre_affinity', 'director_match']):
        df['genre_director_synergy'] = df['genre_affinity'] * df['director_match']
    
    if all(col in df.columns for col in ['actor_overlap', 'keyword_similarity']):
        df['actor_keyword_synergy'] = df['actor_overlap'] * df['keyword_similarity']
    
    return df        