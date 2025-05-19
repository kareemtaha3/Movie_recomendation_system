import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
from .logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    A utility class for validating data throughout the movie recommendation pipeline.
    
    This class provides methods to check data integrity, detect outliers,
    validate schema, and ensure data quality before processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataValidator with optional configuration.
        
        Args:
            config: Optional configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.rating_min = self.config.get('rating_min', 1.0)
        self.rating_max = self.config.get('rating_max', 5.0)
        self.min_ratings_per_user = self.config.get('min_ratings_per_user', 5)
        self.min_ratings_per_movie = self.config.get('min_ratings_per_movie', 5)
        
    def validate_movies_data(self, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate the movies dataframe.
        
        Args:
            movies_df: DataFrame containing movie data
            
        Returns:
            Tuple of (validated_dataframe, validation_report)
        """
        if movies_df is None or movies_df.empty:
            logger.error("Movies dataframe is empty or None")
            raise ValueError("Movies dataframe cannot be empty")
        
        # Check required columns
        required_columns = ['movieId', 'title', 'genres']
        missing_columns = [col for col in required_columns if col not in movies_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in movies dataframe: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate movie IDs
        duplicate_ids = movies_df[movies_df.duplicated('movieId')]['movieId'].tolist()
        
        # Check for missing values
        missing_values = movies_df[required_columns].isnull().sum().to_dict()
        
        # Check for empty genres
        empty_genres = movies_df[movies_df['genres'] == '(no genres listed)'].shape[0]
        
        # Create validation report
        validation_report = {
            'total_movies': len(movies_df),
            'duplicate_movie_ids': duplicate_ids,
            'missing_values': missing_values,
            'empty_genres': empty_genres
        }
        
        # Log validation results
        logger.info(f"Movies data validation report: {validation_report}")
        
        # Remove duplicates if any
        if duplicate_ids:
            logger.warning(f"Removing {len(duplicate_ids)} duplicate movie IDs")
            movies_df = movies_df.drop_duplicates('movieId')
        
        # Fill missing titles if any
        if missing_values['title'] > 0:
            logger.warning(f"Filling {missing_values['title']} missing movie titles")
            movies_df['title'] = movies_df['title'].fillna('Unknown Title')
        
        # Fill missing genres if any
        if missing_values['genres'] > 0:
            logger.warning(f"Filling {missing_values['genres']} missing movie genres")
            movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')
        
        return movies_df, validation_report
    
    def validate_ratings_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate the ratings dataframe.
        
        Args:
            ratings_df: DataFrame containing rating data
            
        Returns:
            Tuple of (validated_dataframe, validation_report)
        """
        if ratings_df is None or ratings_df.empty:
            logger.error("Ratings dataframe is empty or None")
            raise ValueError("Ratings dataframe cannot be empty")
        
        # Check required columns
        required_columns = ['userId', 'movieId', 'rating']
        missing_columns = [col for col in required_columns if col not in ratings_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in ratings dataframe: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_values = ratings_df[required_columns].isnull().sum().to_dict()
        
        # Check for invalid ratings
        invalid_ratings = ratings_df[
            (ratings_df['rating'] < self.rating_min) | 
            (ratings_df['rating'] > self.rating_max)
        ].shape[0]
        
        # Check for users with too few ratings
        user_rating_counts = ratings_df['userId'].value_counts()
        users_with_few_ratings = user_rating_counts[user_rating_counts < self.min_ratings_per_user].shape[0]
        
        # Check for movies with too few ratings
        movie_rating_counts = ratings_df['movieId'].value_counts()
        movies_with_few_ratings = movie_rating_counts[movie_rating_counts < self.min_ratings_per_movie].shape[0]
        
        # Create validation report
        validation_report = {
            'total_ratings': len(ratings_df),
            'missing_values': missing_values,
            'invalid_ratings': invalid_ratings,
            'users_with_few_ratings': users_with_few_ratings,
            'movies_with_few_ratings': movies_with_few_ratings,
            'unique_users': ratings_df['userId'].nunique(),
            'unique_movies': ratings_df['movieId'].nunique(),
            'rating_stats': {
                'min': ratings_df['rating'].min(),
                'max': ratings_df['rating'].max(),
                'mean': ratings_df['rating'].mean(),
                'median': ratings_df['rating'].median(),
                'std': ratings_df['rating'].std()
            }
        }
        
        # Log validation results
        logger.info(f"Ratings data validation report: {validation_report}")
        
        # Remove missing values if any
        if sum(missing_values.values()) > 0:
            logger.warning(f"Removing {sum(missing_values.values())} rows with missing values")
            ratings_df = ratings_df.dropna(subset=required_columns)
        
        # Remove invalid ratings if any
        if invalid_ratings > 0:
            logger.warning(f"Removing {invalid_ratings} rows with invalid ratings")
            ratings_df = ratings_df[
                (ratings_df['rating'] >= self.rating_min) & 
                (ratings_df['rating'] <= self.rating_max)
            ]
        
        return ratings_df, validation_report
    
    def filter_by_interaction_threshold(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ratings to include only users and movies with sufficient interactions.
        
        Args:
            ratings_df: DataFrame containing rating data
            
        Returns:
            Filtered DataFrame
        """
        # Filter users with too few ratings
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        
        # Filter movies with too few ratings
        movie_counts = ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        
        # Apply filters
        filtered_df = ratings_df[
            ratings_df['userId'].isin(valid_users) & 
            ratings_df['movieId'].isin(valid_movies)
        ]
        
        # Log filtering results
        logger.info(f"Filtered ratings from {len(ratings_df)} to {len(filtered_df)} rows")
        logger.info(f"Filtered users from {ratings_df['userId'].nunique()} to {filtered_df['userId'].nunique()}")
        logger.info(f"Filtered movies from {ratings_df['movieId'].nunique()} to {filtered_df['movieId'].nunique()}")
        
        return filtered_df
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in a specific column using various methods.
        
        Args:
            df: DataFrame containing the data
            column: Column name to check for outliers
            method: Method to use for outlier detection ('iqr', 'zscore', 'percentile')
            threshold: Threshold for outlier detection
            
        Returns:
            List of indices of outlier rows
        """
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in dataframe")
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        outlier_indices = []
        
        if method == 'iqr':
            # IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = df[
                (df[column] < lower_bound) | 
                (df[column] > upper_bound)
            ].index.tolist()
            
        elif method == 'zscore':
            # Z-score method
            mean = df[column].mean()
            std = df[column].std()
            
            if std == 0:
                logger.warning(f"Standard deviation is zero for column '{column}', skipping zscore outlier detection")
                return []
            
            z_scores = (df[column] - mean) / std
            outlier_indices = df[abs(z_scores) > threshold].index.tolist()
            
        elif method == 'percentile':
            # Percentile method
            lower_bound = df[column].quantile(threshold / 100)
            upper_bound = df[column].quantile(1 - threshold / 100)
            
            outlier_indices = df[
                (df[column] < lower_bound) | 
                (df[column] > upper_bound)
            ].index.tolist()
            
        else:
            logger.error(f"Unknown outlier detection method: {method}")
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        logger.info(f"Detected {len(outlier_indices)} outliers in column '{column}' using {method} method")
        return outlier_indices
    
    def validate_model_input(self, 
                           movie_features: pd.DataFrame, 
                           user_features: pd.DataFrame, 
                           ratings_df: pd.DataFrame) -> Dict:
        """
        Validate input data for the model.
        
        Args:
            movie_features: DataFrame containing movie features
            user_features: DataFrame containing user features
            ratings_df: DataFrame containing rating data
            
        Returns:
            Validation report dictionary
        """
        # Check that dataframes are not empty
        if movie_features is None or movie_features.empty:
            logger.error("Movie features dataframe is empty or None")
            raise ValueError("Movie features dataframe cannot be empty")
            
        if user_features is None or user_features.empty:
            logger.error("User features dataframe is empty or None")
            raise ValueError("User features dataframe cannot be empty")
            
        if ratings_df is None or ratings_df.empty:
            logger.error("Ratings dataframe is empty or None")
            raise ValueError("Ratings dataframe cannot be empty")
        
        # Check that all users and movies in ratings exist in features
        missing_users = set(ratings_df['userId']) - set(user_features['userId'])
        missing_movies = set(ratings_df['movieId']) - set(movie_features['movieId'])
        
        # Check for missing values in feature dataframes
        movie_missing_values = movie_features.isnull().sum().sum()
        user_missing_values = user_features.isnull().sum().sum()
        
        # Create validation report
        validation_report = {
            'movie_features_shape': movie_features.shape,
            'user_features_shape': user_features.shape,
            'ratings_shape': ratings_df.shape,
            'missing_users_in_features': len(missing_users),
            'missing_movies_in_features': len(missing_movies),
            'movie_features_missing_values': movie_missing_values,
            'user_features_missing_values': user_missing_values
        }
        
        # Log validation results
        logger.info(f"Model input validation report: {validation_report}")
        
        # Raise error if there are missing users or movies
        if missing_users:
            logger.warning(f"Found {len(missing_users)} users in ratings that are missing from user features")
            if len(missing_users) > 10:
                logger.warning(f"First 10 missing users: {list(missing_users)[:10]}")
            else:
                logger.warning(f"Missing users: {list(missing_users)}")
        
        if missing_movies:
            logger.warning(f"Found {len(missing_movies)} movies in ratings that are missing from movie features")
            if len(missing_movies) > 10:
                logger.warning(f"First 10 missing movies: {list(missing_movies)[:10]}")
            else:
                logger.warning(f"Missing movies: {list(missing_movies)}")
        
        return validation_report
    
    def validate_model_output(self, predictions: np.ndarray) -> Dict:
        """
        Validate model predictions.
        
        Args:
            predictions: Array of model predictions
            
        Returns:
            Validation report dictionary
        """
        # Check that predictions are not empty
        if predictions is None or len(predictions) == 0:
            logger.error("Predictions array is empty or None")
            raise ValueError("Predictions array cannot be empty")
        
        # Check for NaN values
        nan_count = np.isnan(predictions).sum()
        
        # Check for out-of-range predictions
        below_min = (predictions < self.rating_min).sum()
        above_max = (predictions > self.rating_max).sum()
        
        # Create validation report
        validation_report = {
            'prediction_count': len(predictions),
            'nan_predictions': nan_count,
            'below_min_predictions': below_min,
            'above_max_predictions': above_max,
            'prediction_stats': {
                'min': np.nanmin(predictions) if not np.all(np.isnan(predictions)) else None,
                'max': np.nanmax(predictions) if not np.all(np.isnan(predictions)) else None,
                'mean': np.nanmean(predictions) if not np.all(np.isnan(predictions)) else None,
                'median': np.nanmedian(predictions) if not np.all(np.isnan(predictions)) else None,
                'std': np.nanstd(predictions) if not np.all(np.isnan(predictions)) else None
            }
        }
        
        # Log validation results
        logger.info(f"Model output validation report: {validation_report}")
        
        # Raise warning if there are NaN or out-of-range predictions
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN predictions")
        
        if below_min > 0:
            logger.warning(f"Found {below_min} predictions below minimum rating {self.rating_min}")
        
        if above_max > 0:
            logger.warning(f"Found {above_max} predictions above maximum rating {self.rating_max}")
        
        return validation_report
    
    def validate_train_test_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Validate train-test split to ensure no data leakage and proper distribution.
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
            
        Returns:
            Validation report dictionary
        """
        # Check that dataframes are not empty
        if train_df is None or train_df.empty:
            logger.error("Training dataframe is empty or None")
            raise ValueError("Training dataframe cannot be empty")
            
        if test_df is None or test_df.empty:
            logger.error("Testing dataframe is empty or None")
            raise ValueError("Testing dataframe cannot be empty")
        
        # Check for overlapping rows
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        overlap = train_indices.intersection(test_indices)
        
        # Check for distribution differences
        train_rating_mean = train_df['rating'].mean() if 'rating' in train_df.columns else None
        test_rating_mean = test_df['rating'].mean() if 'rating' in test_df.columns else None
        
        train_user_count = train_df['userId'].nunique() if 'userId' in train_df.columns else None
        test_user_count = test_df['userId'].nunique() if 'userId' in test_df.columns else None
        
        train_movie_count = train_df['movieId'].nunique() if 'movieId' in train_df.columns else None
        test_movie_count = test_df['movieId'].nunique() if 'movieId' in test_df.columns else None
        
        # Check for users or movies in test set but not in train set
        if 'userId' in train_df.columns and 'userId' in test_df.columns:
            train_users = set(train_df['userId'])
            test_users = set(test_df['userId'])
            users_only_in_test = test_users - train_users
        else:
            users_only_in_test = set()
        
        if 'movieId' in train_df.columns and 'movieId' in test_df.columns:
            train_movies = set(train_df['movieId'])
            test_movies = set(test_df['movieId'])
            movies_only_in_test = test_movies - train_movies
        else:
            movies_only_in_test = set()
        
        # Create validation report
        validation_report = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_test_ratio': len(train_df) / len(test_df) if len(test_df) > 0 else None,
            'overlapping_rows': len(overlap),
            'train_rating_mean': train_rating_mean,
            'test_rating_mean': test_rating_mean,
            'train_user_count': train_user_count,
            'test_user_count': test_user_count,
            'train_movie_count': train_movie_count,
            'test_movie_count': test_movie_count,
            'users_only_in_test': len(users_only_in_test),
            'movies_only_in_test': len(movies_only_in_test)
        }
        
        # Log validation results
        logger.info(f"Train-test split validation report: {validation_report}")
        
        # Raise warning if there are overlapping rows
        if overlap:
            logger.warning(f"Found {len(overlap)} overlapping rows between train and test sets")
        
        # Raise warning if there are users or movies only in test set
        if users_only_in_test:
            logger.warning(f"Found {len(users_only_in_test)} users in test set that are not in train set")
        
        if movies_only_in_test:
            logger.warning(f"Found {len(movies_only_in_test)} movies in test set that are not in train set")
        
        return validation_report