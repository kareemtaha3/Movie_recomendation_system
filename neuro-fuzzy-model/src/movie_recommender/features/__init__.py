"""
Feature Engineering Module for Movie Recommendation System

This module provides functionality for engineering features from movie and user data,
including movie features, user features, and interaction features.

Modules:
    - movie_feature_engineering: Movie-specific feature engineering
    - user_feature_engineering: User-specific feature engineering
    - interaction_feature_engineering: User-movie interaction features
    - feature_engineering: General feature engineering utilities

Example:
    >>> from movie_recommender.features import FeatureEngineering
    >>> fe = FeatureEngineering()
    >>> features = fe.prepare_features_for_model(movies_df, ratings_df)
"""



from .movie_feature_engineering import (
    engineer_movie_features,
    MultiLabelBinarizer
)

from .user_feature_engineering import (
    normalize_numeric_features,
    create_genre_embeddings,
    create_all_features,
    calculate_genre_affinity,
    get_director_match,
    calculate_actor_overlap,
    generate_interaction_features,
    batch_interaction_features,
    create_enhanced_interaction_features
)

from .interaction_feature_engineering import (
    engineer_interaction_features,
    create_enhanced_features
)

__all__ = [
    # Feature Engineering
    'FeatureEngineering',
    
    # Movie Features
    'engineer_movie_features',
    'MultiLabelBinarizer',
    
    # User Features
    'normalize_numeric_features',
    'create_genre_embeddings',
    'calculate_genre_affinity',
    'get_director_match',
    'calculate_actor_overlap',
    'create_enhanced_interaction_features',
    
    # Interaction Features
    'engineer_interaction_features',
    'create_enhanced_features'
]
