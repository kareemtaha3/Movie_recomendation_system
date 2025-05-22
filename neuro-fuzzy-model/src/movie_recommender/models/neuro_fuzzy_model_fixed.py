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
import pandas as pd

from movie_recommender.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class NeuroFuzzyRecommender:
    """
    A neuro-fuzzy recommendation system that combines neural networks with fuzzy logic
    for movie recommendations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the neuro-fuzzy recommender model.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the model.
        """
        self.config = config or {}
        self.neural_model = None
        self.fuzzy_system = None
        self.movie_features = None
        self.user_features = None
        self.history = None
        
        # Set default configuration if not provided
        self._set_default_config()
        
        logger.info("Initialized NeuroFuzzyRecommender with config: %s", self.config)
    
    def _set_default_config(self):
        """
        Set default configuration parameters if not provided.
        """
        defaults = {
            'neural_layers': [64, 32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu',
            'fuzzy_membership_funcs': 3,
            'fuzzy_rules': 'auto',
            'embedding_dim': 32,
            'neural_fuzzy_weight': 0.7
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_neural_network(self, input_dim):
        """
        Build the neural network component of the model.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
            
        Returns
        -------
        keras.Model
            The neural network model.
        """
        logger.info("Building neural network with input dimension: %d", input_dim)
        
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        # Add embedding layer if specified
        if self.config.get('embedding_dim'):
            x = layers.Dense(self.config['embedding_dim'], activation=self.config['activation'])(x)
        
        # Add hidden layers
        for units in self.config['neural_layers']:
            x = layers.Dense(units, activation=self.config['activation'])(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer for rating prediction
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info("Neural network built successfully")
        return model
    
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
        
        # Add rules based on available features
        if 'genre_match' in fuzzy_vars and 'rating_match' in fuzzy_vars:
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_match']['high'] & fuzzy_vars['rating_match']['high'],
                rating['good']
            ))
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_match']['high'] & fuzzy_vars['rating_match']['medium'],
                rating['good']
            ))
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_match']['medium'] & fuzzy_vars['rating_match']['high'],
                rating['good']
            ))
            rules.append(ctrl.Rule(
                fuzzy_vars['genre_match']['low'] & fuzzy_vars['rating_match']['low'],
                rating['poor']
            ))
        
        # Add individual feature rules
        if 'genre_match' in fuzzy_vars:
            rules.append(ctrl.Rule(fuzzy_vars['genre_match']['high'], rating['good']))
            rules.append(ctrl.Rule(fuzzy_vars['genre_match']['low'], rating['poor']))
        
        if 'rating_match' in fuzzy_vars:
            rules.append(ctrl.Rule(fuzzy_vars['rating_match']['high'], rating['good']))
            rules.append(ctrl.Rule(fuzzy_vars['rating_match']['low'], rating['poor']))
        
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
    
    def _prepare_features(self, user_ids, movie_ids):
        """
        Prepare features for the model based on user and movie IDs.
        
        Parameters
        ----------
        user_ids : array-like
            User IDs for which to prepare features.
        movie_ids : array-like
            Movie IDs for which to prepare features.
            
        Returns
        -------
        numpy.ndarray
            Combined features for the model.
        """
        if self.movie_features is None or self.user_features is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Get user features
        user_features_array = []
        for user_id in user_ids:
            user_row = self.user_features[self.user_features['userId'] == user_id]
            if len(user_row) == 0:
                # Use average user features if user not found
                logger.warning(f"User {user_id} not found in training data. Using average features.")
                user_row = self.user_features.mean().to_frame().T
                user_row['userId'] = user_id
            user_features_array.append(user_row.iloc[0].values[1:])
        user_features_array = np.array(user_features_array)
        
        # Get movie features
        movie_features_array = []
        for movie_id in movie_ids:
            movie_row = self.movie_features[self.movie_features['movieId'] == movie_id]
            if len(movie_row) == 0:
                # Use average movie features if movie not found
                logger.warning(f"Movie {movie_id} not found in training data. Using average features.")
                movie_row = self.movie_features.mean().to_frame().T
                movie_row['movieId'] = movie_id
            movie_features_array.append(movie_row.iloc[0].values[1:])
        movie_features_array = np.array(movie_features_array)
        
        # Combine features
        combined_features = np.hstack((user_features_array, movie_features_array))
        
        return combined_features
    
    def _calculate_fuzzy_features(self, user_ids, movie_ids):
        """
        Calculate fuzzy features for user-movie pairs.
        
        Parameters
        ----------
        user_ids : array-like
            User IDs for which to calculate fuzzy features.
        movie_ids : array-like
            Movie IDs for which to calculate fuzzy features.
            
        Returns
        -------
        dict
            Dictionary of fuzzy features.
        """
        # Initialize features
        num_pairs = len(user_ids)
        genre_match = np.zeros(num_pairs)
        rating_match = np.zeros(num_pairs)
        
        # Calculate actual fuzzy features for each user-movie pair
        for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            # Get user and movie data
            user_data = self.user_features[self.user_features['userId'] == user_id]
            movie_data = self.movie_features[self.movie_features['movieId'] == movie_id]
            
            # Handle missing data
            if len(user_data) == 0 or len(movie_data) == 0:
                genre_match[i] = 0.5  # Default to middle value
                rating_match[i] = 0.5
                continue
            
            # Calculate genre match
            # Find genre columns in movie features
            genre_cols = [col for col in self.movie_features.columns if col.startswith('genre_')]
            if genre_cols:
                # Get user's average genre preferences
                user_genre_prefs = user_data[genre_cols].values[0]
                movie_genres = movie_data[genre_cols].values[0]
                
                # Calculate similarity (dot product normalized)
                if np.sum(movie_genres) > 0 and np.sum(user_genre_prefs) > 0:
                    genre_match[i] = np.dot(user_genre_prefs, movie_genres) / (
                        np.sqrt(np.sum(user_genre_prefs**2)) * np.sqrt(np.sum(movie_genres**2)))
                else:
                    genre_match[i] = 0
            
            # Calculate rating match based on user's average rating and movie's average rating
            if 'rating_mean' in user_data.columns and 'rating_mean' in movie_data.columns:
                user_avg_rating = user_data['rating_mean'].values[0]
                movie_avg_rating = movie_data['rating_mean'].values[0]
                
                # Normalize to [0, 1] range assuming ratings are 1-5
                rating_match[i] = 1 - abs(user_avg_rating - movie_avg_rating) / 4
        
        # Ensure values are in [0, 1] range
        genre_match = np.clip(genre_match, 0, 1)
        rating_match = np.clip(rating_match, 0, 1)
        
        return {
            'genre_match': genre_match,
            'rating_match': rating_match
        }
    
    @log_execution_time
    def fit(self, movie_features, user_features, ratings_df, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the neuro-fuzzy recommendation model.
        
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
        logger.info("Training neuro-fuzzy recommendation model")
        
        # Store feature DataFrames for later use
        self.movie_features = movie_features
        self.user_features = user_features
        
        # Prepare training data
        user_ids = ratings_df['userId'].values
        movie_ids = ratings_df['movieId'].values
        ratings = ratings_df['rating'].values
        
        # Prepare features for neural network
        X = self._prepare_features(user_ids, movie_ids)
        y = ratings
        
        # Build neural network
        self.neural_model = self._build_neural_network(X.shape[1])
        
        # Add early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train neural network
        self.history = self.neural_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate fuzzy features
        fuzzy_features = self._calculate_fuzzy_features(user_ids, movie_ids)
        
        # Build fuzzy system
        self.fuzzy_system = self._build_fuzzy_system(list(fuzzy_features.keys()))
        
        logger.info("Model training completed")
        return self
    
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
        if self.neural_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Prepare features for neural network
        X = self._prepare_features(user_ids, movie_ids)
        
        # Get neural network predictions
        nn_predictions = self.neural_model.predict(X).flatten()
        
        # Calculate fuzzy features
        fuzzy_features = self._calculate_fuzzy_features(user_ids, movie_ids)
        
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
        if self.neural_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Get all movie IDs
        all_movie_ids = self.movie_features['movieId'].values
        
        # Exclude movies already rated by the user if requested
        if exclude_rated:
            # Get movies rated by this user
            user_rated_movies = set()
            user_ratings = self.user_features[self.user_features['userId'] == user_id]
            if not user_ratings.empty and 'rated_movies' in user_ratings.columns:
                rated_movies = user_ratings.iloc[0]['rated_movies']
                if rated_movies is not None:
                    user_rated_movies = set(rated_movies)
            
            # Filter out rated movies
            all_movie_ids = np.array([mid for mid in all_movie_ids if mid not in user_rated_movies])
        
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
        if self.neural_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save neural network model
        neural_model_path = os.path.join(model_dir, 'neural_model.h5')
        self.neural_model.save(neural_model_path)
        
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
        
        # Save feature column names for future reference
        feature_info = {
            'movie_columns': list(self.movie_features.columns),
            'user_columns': list(self.user_features.columns)
        }
        feature_info_path = os.path.join(model_dir, 'feature_info.joblib')
        joblib.dump(feature_info, feature_info_path)
        
        # Save training history if available
        if self.history is not None:
            history_path = os.path.join(model_dir, 'training_history.joblib')
            joblib.dump(self.history.history, history_path)
        
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
        NeuroFuzzyRecommender
            The loaded model instance.
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'config.joblib')
        config = joblib.load(config_path)
        
        # Create model instance with loaded configuration
        model = cls(config)
        
        # Load neural network model
        neural_model_path = os.path.join(model_dir, 'neural_model.h5')
        model.neural_model = keras.models.load_model(neural_model_path)
        
        # Load fuzzy system
        fuzzy_system_path = os.path.join(model_dir, 'fuzzy_system.joblib')
        model.fuzzy_system = joblib.load(fuzzy_system_path)
        
        # Recreate control system and simulation
        if model.fuzzy_system['rules']:
            control_system = ctrl.ControlSystem(model.fuzzy_system['rules'])
            simulation = ctrl.ControlSystemSimulation(control_system)
            model.fuzzy_system['control_system'] = control_system
            model.fuzzy_system['simulation'] = simulation
        
        # Load training history if available
        history_path = os.path.join(model_dir, 'training_history.joblib')
        if os.path.exists(history_path):
            history_dict = joblib.load(history_path)
            model.history = type('obj', (object,), {'history': history_dict})
        
        logger.info(f"Model loaded from {model_dir}")
        return model
    
    def plot_training_history(self):
        """
        Plot the training history of the neural network.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss values
        ax1.plot(self.history.history['loss'])
        ax1.plot(self.history.history['val_loss'])
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        ax2.plot(self.history.history['mae'])
        ax2.plot(self.history.history['val_mae'])
        ax2.set_title('Model Mean Absolute Error')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        return fig
    
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