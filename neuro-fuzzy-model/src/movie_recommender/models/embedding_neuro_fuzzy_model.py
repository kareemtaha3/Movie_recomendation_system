import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import logging
import os
import joblib

from movie_recommender.utils.logging import get_logger, log_execution_time
from movie_recommender.models.embedding_model import EmbeddingRecommender

logger = get_logger(__name__)


class EmbeddingNeuroFuzzyRecommender:
    """
    A neuro-fuzzy recommendation system that combines embedding-based neural networks 
    with fuzzy logic for movie recommendations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the embedding neuro-fuzzy recommender model.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the model.
        """
        self.config = config or {}
        self.embedding_model = None
        self.fuzzy_system = None
        self.movie_features = None
        self.user_features = None
        
        # Set default configuration if not provided
        self._set_default_config()
        
        logger.info("Initialized EmbeddingNeuroFuzzyRecommender with config: %s", self.config)
    
    def _set_default_config(self):
        """
        Set default configuration parameters if not provided.
        """
        defaults = {
            'embedding_dim': 32,
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu',
            'l2_regularization': 0.01,
            'fuzzy_membership_funcs': 3,
            'fuzzy_rules': 'auto',
            'neural_fuzzy_weight': 0.7  # Weight for neural network predictions
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_fuzzy_system(self, feature_names):
        """
        Build the fuzzy logic component of the model.
        
        Parameters
        ----------
        feature_names : list
            List of feature names to use in the fuzzy system.
            
        Returns
        -------
        dict
            The fuzzy system components.
        """
        logger.info("Building fuzzy system with features: %s", feature_names)
        
        # Create fuzzy variables for each feature
        fuzzy_vars = {}
        for feature in feature_names:
            fuzzy_vars[feature] = ctrl.Antecedent(np.linspace(0, 1, 100), feature)
            
            # Define membership functions
            if self.config['fuzzy_membership_funcs'] == 3:
                fuzzy_vars[feature]['low'] = fuzz.trimf(fuzzy_vars[feature].universe, [0, 0, 0.5])
                fuzzy_vars[feature]['medium'] = fuzz.trimf(fuzzy_vars[feature].universe, [0, 0.5, 1])
                fuzzy_vars[feature]['high'] = fuzz.trimf(fuzzy_vars[feature].universe, [0.5, 1, 1])
            elif self.config['fuzzy_membership_funcs'] == 5:
                fuzzy_vars[feature]['very_low'] = fuzz.trimf(fuzzy_vars[feature].universe, [0, 0, 0.25])
                fuzzy_vars[feature]['low'] = fuzz.trimf(fuzzy_vars[feature].universe, [0, 0.25, 0.5])
                fuzzy_vars[feature]['medium'] = fuzz.trimf(fuzzy_vars[feature].universe, [0.25, 0.5, 0.75])
                fuzzy_vars[feature]['high'] = fuzz.trimf(fuzzy_vars[feature].universe, [0.5, 0.75, 1])
                fuzzy_vars[feature]['very_high'] = fuzz.trimf(fuzzy_vars[feature].universe, [0.75, 1, 1])
        
        # Create fuzzy output variable
        rating = ctrl.Consequent(np.linspace(1, 5, 100), 'rating')
        
        # Define membership functions for rating
        rating['poor'] = fuzz.trimf(rating.universe, [1, 1, 3])
        rating['average'] = fuzz.trimf(rating.universe, [1, 3, 5])
        rating['good'] = fuzz.trimf(rating.universe, [3, 5, 5])
        
        # Create fuzzy rules
        rules = []
        
        # Add rules based on embedding similarity
        if 'embedding_similarity' in fuzzy_vars:
            rules.append(ctrl.Rule(
                fuzzy_vars['embedding_similarity']['high'],
                rating['good']
            ))
            rules.append(ctrl.Rule(
                fuzzy_vars['embedding_similarity']['low'],
                rating['poor']
            ))
        
        # Add rules based on genre affinity
        if 'genre_affinity' in fuzzy_vars:
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_affinity']['high'],
                rating['good']
            ))
        
        # Add rules based on combined features
        if 'genre_affinity' in fuzzy_vars and 'embedding_similarity' in fuzzy_vars:
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_affinity']['high'] & fuzzy_vars['embedding_similarity']['high'],
                rating['good']
            ))
        
        # Create control system
        if rules:
            control_system = ctrl.ControlSystem(rules)
            simulation = ctrl.ControlSystemSimulation(control_system)
        else:
            control_system = None
            simulation = None
            logger.warning("No fuzzy rules created. Fuzzy system will not be functional.")
        
        fuzzy_system = {
            'variables': fuzzy_vars,
            'output': rating,
            'rules': rules,
            'control_system': control_system,
            'simulation': simulation
        }
        
        logger.info("Fuzzy system built successfully with %d rules", len(rules))
        return fuzzy_system
    
    @log_execution_time
    def fit(self, movie_features, user_features, ratings_df, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the embedding neuro-fuzzy recommendation model.
        
        Parameters
        ----------
        movie_features : pandas.DataFrame
            DataFrame containing movie features.
        user_features : pandas.DataFrame
            DataFrame containing user features.
        ratings_df : pandas.DataFrame
            DataFrame containing user-movie ratings.
        epochs : int, optional
            Number of training epochs. Default is 10.
        batch_size : int, optional
            Batch size for training. Default is 32.
        validation_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
            
        Returns
        -------
        self
            The trained model instance.
        """
        logger.info("Training embedding neuro-fuzzy recommendation model")
        
        # Store feature DataFrames for later use
        self.movie_features = movie_features
        self.user_features = user_features
        
        # Create and train embedding model
        self.embedding_model = EmbeddingRecommender({
            'embedding_dim': self.config['embedding_dim'],
            'hidden_layers': self.config['hidden_layers'],
            'dropout_rate': self.config['dropout_rate'],
            'learning_rate': self.config['learning_rate'],
            'activation': self.config['activation'],
            'l2_regularization': self.config['l2_regularization']
        })
        
        # Train embedding model
        self.embedding_model, history = self.embedding_model.fit(
            user_features, 
            movie_features, 
            ratings_df,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # Get embeddings for all users and movies
        embeddings = self.embedding_model.get_embeddings(user_features, movie_features)
        
        # Calculate fuzzy features using embeddings
        fuzzy_features = self._calculate_fuzzy_features_from_embeddings(
            embeddings['user_embeddings']['embeddings'],
            embeddings['movie_embeddings']['embeddings'],
            ratings_df['userId'].values,
            ratings_df['movieId'].values,
            user_features,
            movie_features
        )
        
        # Build fuzzy system
        self.fuzzy_system = self._build_fuzzy_system(list(fuzzy_features.keys()))
        
        logger.info("Model training completed")
        return self, history
    
    def _calculate_fuzzy_features_from_embeddings(self, user_embeddings, movie_embeddings, user_ids, movie_ids, user_features_df, movie_features_df):
        """
        Calculate fuzzy features from embeddings for user-movie pairs.
        
        Parameters
        ----------
        user_embeddings : numpy.ndarray
            User embedding vectors.
        movie_embeddings : numpy.ndarray
            Movie embedding vectors.
        user_ids : array-like
            User IDs for which to calculate fuzzy features.
        movie_ids : array-like
            Movie IDs for which to calculate fuzzy features.
        user_features_df : pandas.DataFrame
            DataFrame containing user features.
        movie_features_df : pandas.DataFrame
            DataFrame containing movie features.
            
        Returns
        -------
        dict
            Dictionary of fuzzy features.
        """
        # Create mapping from IDs to indices
        user_id_to_idx = {user_id: i for i, user_id in enumerate(user_features_df['userId'].values)}
        movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_features_df['movieId'].values)}
        
        # Calculate embedding similarity (cosine similarity)
        embedding_similarity = np.zeros(len(user_ids))
        for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            user_idx = user_id_to_idx.get(user_id)
            movie_idx = movie_id_to_idx.get(movie_id)
            
            if user_idx is not None and movie_idx is not None:
                user_emb = user_embeddings[user_idx]
                movie_emb = movie_embeddings[movie_idx]
                
                # Compute cosine similarity
                similarity = np.dot(user_emb, movie_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(movie_emb))
                embedding_similarity[i] = (similarity + 1) / 2  # Scale from [-1, 1] to [0, 1]
        
        # Extract genre features if available
        genre_affinity = None
        genre_cols = [col for col in movie_features_df.columns if col.startswith('genre_')]
        if genre_cols:
            genre_affinity = np.zeros(len(user_ids))
            for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
                user_row = user_features_df[user_features_df['userId'] == user_id]
                movie_row = movie_features_df[movie_features_df['movieId'] == movie_id]
                
                if not user_row.empty and not movie_row.empty:
                    # Get user genre preferences (assuming they're in columns like user_genre_pref_*)
                    user_genre_cols = [f'user_genre_pref_{g.split("_", 1)[1]}' for g in genre_cols]
                    if all(col in user_row.columns for col in user_genre_cols):
                        user_genres = user_row[user_genre_cols].values[0]
                        movie_genres = movie_row[genre_cols].values[0]
                        
                        # Calculate dot product as genre affinity
                        genre_affinity[i] = np.dot(user_genres, movie_genres)
        
        # Create dictionary of fuzzy features
        fuzzy_features = {'embedding_similarity': embedding_similarity}
        if genre_affinity is not None:
            fuzzy_features['genre_affinity'] = genre_affinity
        
        return fuzzy_features
    
    @log_execution_time
    def predict(self, user_ids, movie_ids):
        """
        Predict ratings for user-movie pairs.
        
        Parameters
        ----------
        user_ids : array-like
            User IDs for which to predict ratings.
        movie_ids : array-like
            Movie IDs for which to predict ratings.
            
        Returns
        -------
        numpy.ndarray
            Predicted ratings.
        """
        if self.embedding_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Prepare features for neural network
        user_features_array = np.array([
            self.user_features.loc[self.user_features['userId'] == user_id].iloc[0].values[1:] 
            for user_id in user_ids
        ])
        
        movie_features_array = np.array([
            self.movie_features.loc[self.movie_features['movieId'] == movie_id].iloc[0].values[1:] 
            for movie_id in movie_ids
        ])
        
        # Get neural network predictions
        nn_predictions = self.embedding_model.predict(user_features_array, movie_features_array)
        
        # Get embeddings for all users and movies
        user_embeddings = self.embedding_model.get_embeddings(self.user_features)['user_embeddings']['embeddings']
        movie_embeddings = self.embedding_model.get_embeddings(movie_features=self.movie_features)['movie_embeddings']['embeddings']
        
        # Calculate fuzzy features
        fuzzy_features = self._calculate_fuzzy_features_from_embeddings(
            user_embeddings,
            movie_embeddings,
            user_ids,
            movie_ids,
            self.user_features,
            self.movie_features
        )
        
        # Apply fuzzy rules if fuzzy system is available
        fuzzy_predictions = np.zeros_like(nn_predictions)
        if self.fuzzy_system['simulation'] is not None:
            for i in range(len(user_ids)):
                try:
                    # Set input values for fuzzy system
                    for feature, values in fuzzy_features.items():
                        self.fuzzy_system['simulation'].input[feature] = values[i]
                    
                    # Compute fuzzy output
                    self.fuzzy_system['simulation'].compute()
                    fuzzy_predictions[i] = self.fuzzy_system['simulation'].output['rating']
                except Exception as e:
                    logger.warning(f"Fuzzy prediction failed for user {user_ids[i]} and movie {movie_ids[i]}: {e}")
                    fuzzy_predictions[i] = 3.0  # Default to middle rating
        
        # Combine neural network and fuzzy predictions
        alpha = self.config.get('neural_fuzzy_weight', 0.7)  # Weight for neural network predictions
        combined_predictions = alpha * nn_predictions + (1 - alpha) * fuzzy_predictions
        
        # Clip predictions to valid rating range
        combined_predictions = np.clip(combined_predictions, 1.0, 5.0)
        
        return combined_predictions
    
    def recommend_movies(self, user_id, n=10, exclude_rated=True):
        """
        Recommend top N movies for a user.
        
        Parameters
        ----------
        user_id : int
            User ID for which to recommend movies.
        n : int, optional
            Number of movies to recommend. Default is 10.
        exclude_rated : bool, optional
            Whether to exclude movies already rated by the user. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing recommended movies with predicted ratings.
        """
        if self.embedding_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Get all movie IDs
        all_movie_ids = self.movie_features['movieId'].values
        
        # Exclude movies already rated by the user if requested
        if exclude_rated:
            # Get movies rated by the user
            rated_movie_ids = self.user_features.loc[self.user_features['userId'] == user_id, 'rated_movies']
            if not rated_movie_ids.empty and rated_movie_ids.iloc[0] is not None:
                rated_movie_ids = rated_movie_ids.iloc[0]
                all_movie_ids = np.setdiff1d(all_movie_ids, rated_movie_ids)
        
        # Create arrays of user IDs and movie IDs for prediction
        user_ids = np.full_like(all_movie_ids, user_id)
        movie_ids = all_movie_ids
        
        # Predict ratings
        predicted_ratings = self.predict(user_ids, movie_ids)
        
        # Create DataFrame with movie IDs and predicted ratings
        recommendations = pd.DataFrame({
            'movieId': movie_ids,
            'predicted_rating': predicted_ratings
        })
        
        # Sort by predicted rating in descending order
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        # Get top N recommendations
        top_recommendations = recommendations.head(n)
        
        # Merge with movie features to get movie titles and other information
        top_recommendations = pd.merge(
            top_recommendations,
            self.movie_features[['movieId', 'title', 'genres']],
            on='movieId'
        )
        
        return top_recommendations
    
    def save_model(self, model_dir):
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        model_dir : str
            Directory where to save the model.
            
        Returns
        -------
        str
            Path to the saved model directory.
        """
        if self.embedding_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save embedding model
        embedding_model_dir = os.path.join(model_dir, 'embedding_model')
        self.embedding_model.save_model(embedding_model_dir)
        
        # Save fuzzy system (excluding control system and simulation which are not serializable)
        fuzzy_system_serializable = {
            key: value for key, value in self.fuzzy_system.items() 
            if key not in ['control_system', 'simulation']
        }
        fuzzy_system_path = os.path.join(model_dir, 'fuzzy_system.joblib')
        joblib.dump(fuzzy_system_serializable, fuzzy_system_path)
        
        # Save configuration
        config_path = os.path.join(model_dir, 'config.joblib')
        joblib.dump(self.config, config_path)
        
        logger.info(f"Model saved to {model_dir}")
        return model_dir
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        model_dir : str
            Directory where the model is saved.
            
        Returns
        -------
        EmbeddingNeuroFuzzyRecommender
            The loaded model instance.
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'config.joblib')
        config = joblib.load(config_path)
        
        # Create model instance with loaded configuration
        model = cls(config)
        
        # Load embedding model
        embedding_model_dir = os.path.join(model_dir, 'embedding_model')
        model.embedding_model = EmbeddingRecommender.load_model(embedding_model_dir)
        
        # Load fuzzy system
        fuzzy_system_path = os.path.join(model_dir, 'fuzzy_system.joblib')
        model.fuzzy_system = joblib.load(fuzzy_system_path)
        
        # Recreate control system and simulation
        if model.fuzzy_system['rules']:
            control_system = ctrl.ControlSystem(model.fuzzy_system['rules'])
            simulation = ctrl.ControlSystemSimulation(control_system)
            model.fuzzy_system['control_system'] = control_system
            model.fuzzy_system['simulation'] = simulation
        
        logger.info(f"Model loaded from {model_dir}")
        return model
    
    def plot_fuzzy_membership(self):
        """
        Plot the fuzzy membership functions.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.fuzzy_system is None:
            raise ValueError("Fuzzy system has not been created. Call fit() first.")
        
        # Get fuzzy variables
        variables = list(self.fuzzy_system['variables'].values())
        variables.append(self.fuzzy_system['output'])
        
        # Create figure with subplots for each variable
        n_vars = len(variables)
        fig, axes = plt.subplots(1, n_vars, figsize=(15, 5))
        
        # If there's only one variable, axes is not a list
        if n_vars == 1:
            axes = [axes]
        
        # Plot membership functions for each variable
        for i, var in enumerate(variables):
            var.view(ax=axes[i])
            axes[i].set_title(var.label)
            axes[i].set_ylabel('Membership')
            axes[i].set_xlabel('Value')
        
        plt.tight_layout()
        return fig