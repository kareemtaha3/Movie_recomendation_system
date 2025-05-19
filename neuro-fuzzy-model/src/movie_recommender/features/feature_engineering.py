import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from movie_recommender.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Class for feature engineering operations on movie and user data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering class.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for feature engineering.
        """
        self.config = config or {}
        self.genre_encoder = None
        self.year_scaler = None
        self.tfidf_vectorizer = None
        
    @log_execution_time
    def extract_year_from_title(self, movies_df):
        """
        Extract the release year from movie titles.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with a 'title' column.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added 'year' and 'clean_title' columns.
        """
        logger.info("Extracting year from movie titles")
        
        # Create a copy to avoid modifying the original DataFrame
        df = movies_df.copy()
        
        # Extract year from title using regex
        df['year'] = df['title'].str.extract(r'\((\d{4})\)$')
        
        # Clean title by removing the year part
        df['clean_title'] = df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Convert year to numeric
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Fill missing years with median
        median_year = df['year'].median()
        df['year'].fillna(median_year, inplace=True)
        
        logger.info(f"Extracted years from {len(df)} movie titles")
        return df
    
    @log_execution_time
    def encode_genres(self, movies_df, max_genres=None):
        """
        One-hot encode movie genres.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with a 'genres' column.
        max_genres : int, optional
            Maximum number of genres to include. If None, all genres are included.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added genre columns.
        """
        logger.info("Encoding movie genres")
        
        # Create a copy to avoid modifying the original DataFrame
        df = movies_df.copy()
        
        # Split genres string into a list
        df['genres_list'] = df['genres'].str.split('|')
        
        # Get all unique genres
        all_genres = []
        for genres in df['genres_list']:
            if isinstance(genres, list):
                all_genres.extend(genres)
        
        unique_genres = pd.Series(all_genres).value_counts()
        
        # Limit to max_genres if specified
        if max_genres is not None and max_genres < len(unique_genres):
            top_genres = unique_genres.index[:max_genres]
        else:
            top_genres = unique_genres.index
        
        # Create one-hot encoded columns for each genre
        for genre in top_genres:
            df[f'genre_{genre}'] = df['genres_list'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
        
        # Store the encoder information for future use
        self.genre_encoder = {genre: f'genre_{genre}' for genre in top_genres}
        
        logger.info(f"Encoded {len(top_genres)} genres")
        return df
    
    @log_execution_time
    def normalize_year(self, movies_df):
        """
        Normalize the movie release year.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with a 'year' column.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with normalized 'year_norm' column.
        """
        logger.info("Normalizing movie release years")
        
        # Create a copy to avoid modifying the original DataFrame
        df = movies_df.copy()
        
        # Initialize scaler if not already done
        if self.year_scaler is None:
            self.year_scaler = MinMaxScaler()
            df['year_norm'] = self.year_scaler.fit_transform(df[['year']])
        else:
            df['year_norm'] = self.year_scaler.transform(df[['year']])
        
        logger.info("Normalized movie release years")
        return df
    
    @log_execution_time
    def extract_title_features(self, movies_df, max_features=100):
        """
        Extract features from movie titles using TF-IDF.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with a 'clean_title' column.
        max_features : int, optional
            Maximum number of features to extract. Default is 100.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added title feature columns.
        """
        logger.info(f"Extracting title features with max_features={max_features}")
        
        # Create a copy to avoid modifying the original DataFrame
        df = movies_df.copy()
        
        # Initialize vectorizer if not already done
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            title_features = self.tfidf_vectorizer.fit_transform(df['clean_title'])
        else:
            title_features = self.tfidf_vectorizer.transform(df['clean_title'])
        
        # Convert to DataFrame
        feature_names = [f'title_{i}' for i in range(title_features.shape[1])]
        title_features_df = pd.DataFrame(
            title_features.toarray(),
            columns=feature_names,
            index=df.index
        )
        
        # Concatenate with original DataFrame
        result_df = pd.concat([df, title_features_df], axis=1)
        
        logger.info(f"Extracted {len(feature_names)} title features")
        return result_df
    
    @log_execution_time
    def calculate_user_features(self, ratings_df):
        """
        Calculate user features from ratings data.
        
        Parameters
        ----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with user features.
        """
        logger.info("Calculating user features")
        
        # Group by user and calculate statistics
        user_features = ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Reset index to make userId a column
        user_features.reset_index(inplace=True)
        
        # Fill NaN values in std with 0 (for users with only one rating)
        user_features['rating_std'].fillna(0, inplace=True)
        
        logger.info(f"Calculated features for {len(user_features)} users")
        return user_features
    
    @log_execution_time
    def calculate_movie_features(self, ratings_df):
        """
        Calculate movie features from ratings data.
        
        Parameters
        ----------
        ratings_df : pandas.DataFrame
            DataFrame containing movie ratings.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with movie features.
        """
        logger.info("Calculating movie features")
        
        # Group by movie and calculate statistics
        movie_features = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        movie_features.columns = ['_'.join(col).strip() for col in movie_features.columns.values]
        
        # Reset index to make movieId a column
        movie_features.reset_index(inplace=True)
        
        # Fill NaN values in std with 0 (for movies with only one rating)
        movie_features['rating_std'].fillna(0, inplace=True)
        
        logger.info(f"Calculated features for {len(movie_features)} movies")
        return movie_features
    
    @log_execution_time
    def prepare_features_for_model(self, movies_df, ratings_df, users_df=None):
        """
        Prepare all features for the recommendation model.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information.
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings.
        users_df : pandas.DataFrame, optional
            DataFrame containing user information.
            
        Returns
        -------
        tuple
            Tuple containing (movie_features, user_features, ratings).
        """
        logger.info("Preparing all features for the recommendation model")
        
        # Process movie features
        movies_with_year = self.extract_year_from_title(movies_df)
        movies_with_genres = self.encode_genres(
            movies_with_year, 
            max_genres=self.config.get('max_genres', 20)
        )
        movies_normalized = self.normalize_year(movies_with_genres)
        
        # Calculate movie statistics from ratings
        movie_stats = self.calculate_movie_features(ratings_df)
        
        # Merge movie features
        movie_features = pd.merge(
            movies_normalized,
            movie_stats,
            on='movieId',
            how='left'
        )
        
        # Fill missing rating statistics with defaults
        movie_features['rating_count'].fillna(0, inplace=True)
        movie_features['rating_mean'].fillna(movie_features['rating_mean'].mean(), inplace=True)
        movie_features['rating_std'].fillna(0, inplace=True)
        movie_features['rating_min'].fillna(movie_features['rating_mean'], inplace=True)
        movie_features['rating_max'].fillna(movie_features['rating_mean'], inplace=True)
        
        # Calculate user features
        user_features = self.calculate_user_features(ratings_df)
        
        # If user information is provided, merge it
        if users_df is not None:
            user_features = pd.merge(
                user_features,
                users_df,
                on='userId',
                how='left'
            )
        
        logger.info("Completed feature preparation")
        return movie_features, user_features, ratings_df