import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import logging

logger = logging.getLogger(__name__)

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

def create_user_id_embedding(df, user_id_col='userId', embedding_dim=8):
    """
    Create user ID embeddings using a simple hash-based approach
    
    Args:
        df: DataFrame containing user IDs
        user_id_col: Column name for user IDs
        embedding_dim: Dimension of the embedding
        
    Returns:
        DataFrame with user ID embedding features
    """
    df_copy = df.copy()
    
    # Convert user IDs to numeric if they aren't already
    user_ids = pd.to_numeric(df[user_id_col], errors='coerce')
    
    # Create embeddings using modulo hashing
    for i in range(embedding_dim):
        # Use different prime numbers for each dimension to reduce collisions
        prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][i % 12]
        df_copy[f'user_id_embedding_{i+1}'] = ((user_ids * prime) % 97) / 97
    
    return df_copy

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
    
    # Create user ID embeddings
    if 'userId' in df.columns:
        df = create_user_id_embedding(df)
    
    # Create genre embeddings if we have enough genre columns
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

def create_user_id_embedding(user_id, embedding_size=16):
    """
    Create a simple embedding for user ID using a hash-based approach.
    
    Parameters:
    - user_id: User identifier
    - embedding_size: Size of the embedding vector
    
    Returns:
    - Numpy array with the embedding
    """
    # Convert user_id to string and encode
    user_id_str = str(user_id).encode('utf-8')
    
    # Create a hash of the user_id
    hash_obj = hashlib.md5(user_id_str)
    hash_bytes = hash_obj.digest()
    
    # Convert hash bytes to a list of integers
    hash_ints = [b for b in hash_bytes]
    
    # Ensure we have enough values by cycling if needed
    while len(hash_ints) < embedding_size:
        hash_ints.extend(hash_ints)
    
    # Take only what we need and normalize to range [-1, 1]
    embedding = np.array(hash_ints[:embedding_size]) / 127.5 - 1.0
    
    return embedding

def calculate_user_statistics(user_ratings):
    """
    Calculate basic statistics for a user's ratings.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings
    
    Returns:
    - Dictionary with user statistics
    """
    stats = {}
    
    # Average rating
    stats['user_avg_rating'] = user_ratings['rating'].mean()
    
    # Rating count
    stats['user_rating_count'] = len(user_ratings)
    
    # Rating variance
    stats['user_rating_variance'] = user_ratings['rating'].var()
    
    # Average days to watch (if timestamp and release_date are available)
    if 'timestamp' in user_ratings.columns and 'release_date' in user_ratings.columns:
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(user_ratings['timestamp']):
            user_ratings['timestamp'] = pd.to_datetime(user_ratings['timestamp'], unit='s')
        
        # Convert release_date to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(user_ratings['release_date']):
            user_ratings['release_date'] = pd.to_datetime(user_ratings['release_date'])
        
        # Calculate days between release and rating
        user_ratings['days_to_watch'] = (user_ratings['timestamp'] - user_ratings['release_date']).dt.days
        
        # Average days to watch (only positive values)
        valid_days = user_ratings['days_to_watch'][user_ratings['days_to_watch'] > 0]
        stats['user_avg_days_to_watch'] = valid_days.mean() if not valid_days.empty else np.nan
    else:
        stats['user_avg_days_to_watch'] = np.nan
    
    return stats

def calculate_genre_proportions(user_ratings):
    """
    Calculate the proportion of movies in each genre that the user has rated.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings and movie genres
    
    Returns:
    - Dictionary with genre proportions
    """
    genre_props = {}
    
    # Check if genres column exists
    if 'genres' not in user_ratings.columns:
        return genre_props
    
    # Extract all genres from all movies
    all_genres = []
    for genres_str in user_ratings['genres'].dropna():
        genres = genres_str.split('|')
        all_genres.extend(genres)
    
    # Count occurrences of each genre
    genre_counts = Counter(all_genres)
    total_movies = len(user_ratings)
    
    # Calculate proportion for each genre
    for genre, count in genre_counts.items():
        if genre:  # Skip empty genre
            genre_key = f'user_genre_proportion_{genre.upper()}'
            genre_props[genre_key] = count / total_movies
    
    return genre_props

def calculate_director_preferences(user_ratings):
    """
    Calculate the proportion of movies by each director that the user has rated.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings and movie crew information
    
    Returns:
    - Dictionary with director preferences
    """
    director_prefs = {}
    
    # Check if crew column exists
    if 'crew' not in user_ratings.columns:
        return director_prefs
    
    # Count occurrences of each director
    director_counts = Counter(user_ratings['crew'].dropna())
    total_movies = len(user_ratings)
    
    # Calculate proportion for top directors (limit to top 10 to avoid too many features)
    top_directors = director_counts.most_common(10)
    for director, count in top_directors:
        if director:  # Skip empty director
            director_key = f'user_director_preference_{director.replace(" ", "_").upper()}'
            director_prefs[director_key] = count / total_movies
    
    return director_prefs

def calculate_actor_preferences(user_ratings):
    """
    Calculate the proportion of movies with each actor that the user has rated.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings and movie cast information
    
    Returns:
    - Dictionary with actor preferences
    """
    actor_prefs = {}
    
    # Check if cast column exists
    if 'cast' not in user_ratings.columns:
        return actor_prefs
    
    # Extract all actors from all movies
    all_actors = []
    for cast_str in user_ratings['cast'].dropna():
        actors = cast_str.split('|')
        # Take only the first 3 actors from each movie (usually the main ones)
        all_actors.extend(actors[:3])
    
    # Count occurrences of each actor
    actor_counts = Counter(all_actors)
    total_movies = len(user_ratings)
    
    # Calculate proportion for top actors (limit to top 10 to avoid too many features)
    top_actors = actor_counts.most_common(10)
    for actor, count in top_actors:
        if actor:  # Skip empty actor
            actor_key = f'user_actor_preference_{actor.replace(" ", "_").upper()}'
            actor_prefs[actor_key] = count / total_movies
    
    return actor_prefs

def calculate_release_year_stats(user_ratings):
    """
    Calculate statistics about release years of movies the user liked and disliked.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings and movie release dates
    
    Returns:
    - Dictionary with release year statistics
    """
    year_stats = {}
    
    # Check if release_date column exists
    if 'release_date' not in user_ratings.columns:
        return year_stats
    
    # Convert release_date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(user_ratings['release_date']):
        user_ratings['release_date'] = pd.to_datetime(user_ratings['release_date'])
    
    # Extract year from release_date
    user_ratings['release_year'] = user_ratings['release_date'].dt.year
    
    # Define liked and disliked movies (threshold at 3.5)
    liked_movies = user_ratings[user_ratings['rating'] >= 3.5]
    disliked_movies = user_ratings[user_ratings['rating'] < 3.5]
    
    # Calculate average release year for liked movies
    if not liked_movies.empty:
        year_stats['user_avg_release_year_liked'] = liked_movies['release_year'].mean()
    else:
        year_stats['user_avg_release_year_liked'] = np.nan
    
    # Calculate average release year for disliked movies
    if not disliked_movies.empty:
        year_stats['user_avg_release_year_disliked'] = disliked_movies['release_year'].mean()
    else:
        year_stats['user_avg_release_year_disliked'] = np.nan
    
    return year_stats

def calculate_budget_revenue_stats(user_ratings):
    """
    Calculate statistics about budget and revenue of movies the user liked.
    
    Parameters:
    - user_ratings: DataFrame with user's ratings, movie budgets and revenues
    
    Returns:
    - Dictionary with budget and revenue statistics
    """
    budget_revenue_stats = {}
    
    # Check if budget and revenue columns exist
    if 'budget' not in user_ratings.columns or 'revenue' not in user_ratings.columns:
        return budget_revenue_stats
    
    # Define liked movies (threshold at 3.5)
    liked_movies = user_ratings[user_ratings['rating'] >= 3.5]
    
    # Calculate log budget and revenue to handle skewed distributions
    if not liked_movies.empty:
        # Handle zero or negative values before taking log
        valid_budget = liked_movies['budget'][liked_movies['budget'] > 0]
        valid_revenue = liked_movies['revenue'][liked_movies['revenue'] > 0]
        
        if not valid_budget.empty:
            budget_revenue_stats['user_avg_log_budget_liked'] = np.log1p(valid_budget).mean()
        else:
            budget_revenue_stats['user_avg_log_budget_liked'] = np.nan
        
        if not valid_revenue.empty:
            budget_revenue_stats['user_avg_log_revenue_liked'] = np.log1p(valid_revenue).mean()
        else:
            budget_revenue_stats['user_avg_log_revenue_liked'] = np.nan
    else:
        budget_revenue_stats['user_avg_log_budget_liked'] = np.nan
        budget_revenue_stats['user_avg_log_revenue_liked'] = np.nan
    
    return budget_revenue_stats

def engineer_user_features(ratings_df, movies_df, embedding_size=16):
    """
    Engineer user features based on their rating history.
    
    Parameters:
    - ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
    - movies_df: DataFrame with movie features
    - embedding_size: Size of the user ID embedding vector
    
    Returns:
    - DataFrame with user features
    """
    start_time = time.time()
    logger.info("Starting user feature engineering")
    
    # Merge ratings with movie features
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
    
    # Process each user
    user_features_list = []
    
    for user_id, user_data in tqdm(merged_df.groupby('userId'), desc="Processing users"):
        try:
            user_features = {'userId': user_id}
            
            # Create user ID embedding
            embedding = create_user_id_embedding(user_id, embedding_size)
            for i, val in enumerate(embedding):
                user_features[f'user_id_embedding_{i}'] = val
            
            # Calculate basic statistics
            user_stats = calculate_user_statistics(user_data)
            user_features.update(user_stats)
            
            # Calculate genre proportions
            genre_props = calculate_genre_proportions(user_data)
            user_features.update(genre_props)
            
            # Calculate director preferences
            director_prefs = calculate_director_preferences(user_data)
            user_features.update(director_prefs)
            
            # Calculate actor preferences
            actor_prefs = calculate_actor_preferences(user_data)
            user_features.update(actor_prefs)
            
            # Calculate release year statistics
            year_stats = calculate_release_year_stats(user_data)
            user_features.update(year_stats)
            
            # Calculate budget and revenue statistics
            budget_revenue_stats = calculate_budget_revenue_stats(user_data)
            user_features.update(budget_revenue_stats)
            
            user_features_list.append(user_features)
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
    
    # Create DataFrame from user features
    user_features_df = pd.DataFrame(user_features_list)
    
    # Normalize numeric features
    numeric_cols = user_features_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'userId':
            # Skip columns with all NaN values
            if not user_features_df[col].isna().all():
                # Fill NaN values with mean
                mean_val = user_features_df[col].mean()
                user_features_df[col].fillna(mean_val, inplace=True)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed user feature engineering with {len(user_features_df)} users in {elapsed_time:.2f} seconds")
    
    return user_features_df