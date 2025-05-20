import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

from movie_recommender.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class EmbeddingRecommender:
    """
    An embedding-based recommendation model that creates dense representations
    of users and movies for recommendation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the embedding recommender model.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the model.
        """
        self.config = config or {}
        self.model = None
        self.user_embedding_model = None
        self.movie_embedding_model = None
        
        # Set default configuration if not provided
        self._set_default_config()
        
        logger.info("Initialized EmbeddingRecommender with config: %s", self.config)
    
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
            'l2_regularization': 0.01
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_embedding_model(self, input_dim, name):
        """
        Build an embedding model for users or movies.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        name : str
            Name of the model ('user' or 'movie').
            
        Returns
        -------
        keras.Model
            The embedding model.
        """
        inputs = keras.Input(shape=(input_dim,), name=f"{name}_features")
        x = inputs
        
        # Add hidden layers with regularization
        for i, units in enumerate(self.config['hidden_layers']):
            x = layers.Dense(
                units, 
                activation=self.config['activation'],
                kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                name=f"{name}_dense_{i+1}"
            )(x)
            x = layers.Dropout(self.config['dropout_rate'], name=f"{name}_dropout_{i+1}")(x)
        
        # Final embedding layer
        embedding = layers.Dense(
            self.config['embedding_dim'], 
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
            name=f"{name}_embedding"
        )(x)
        
        # Normalize embeddings to unit length
        embedding_normalized = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name=f"{name}_embedding_normalized"
        )(embedding)
        
        model = keras.Model(inputs=inputs, outputs=[embedding, embedding_normalized], name=f"{name}_embedding_model")
        return model
    
    def _build_recommendation_model(self, user_input_dim, movie_input_dim):
        """
        Build the complete recommendation model using embeddings.
        
        Parameters
        ----------
        user_input_dim : int
            Dimension of the user input features.
        movie_input_dim : int
            Dimension of the movie input features.
            
        Returns
        -------
        keras.Model
            The complete recommendation model.
        """
        # Build user and movie embedding models
        self.user_embedding_model = self._build_embedding_model(user_input_dim, "user")
        self.movie_embedding_model = self._build_embedding_model(movie_input_dim, "movie")
        
        # User and movie inputs
        user_inputs = keras.Input(shape=(user_input_dim,), name="user_features")
        movie_inputs = keras.Input(shape=(movie_input_dim,), name="movie_features")
        
        # Get embeddings
        user_embedding, user_embedding_norm = self.user_embedding_model(user_inputs)
        movie_embedding, movie_embedding_norm = self.movie_embedding_model(movie_inputs)
        
        # Compute dot product similarity
        dot_product = layers.Dot(axes=1, normalize=True, name="dot_product")(
            [user_embedding_norm, movie_embedding_norm]
        )
        
        # Concatenate embeddings and similarity for final prediction
        concat = layers.Concatenate(name="concat")(
            [user_embedding, movie_embedding, dot_product]
        )
        
        # Final prediction layers
        x = layers.Dense(32, activation=self.config['activation'], name="final_dense_1")(concat)
        x = layers.Dropout(self.config['dropout_rate'], name="final_dropout_1")(x)
        x = layers.Dense(16, activation=self.config['activation'], name="final_dense_2")(x)
        
        # Output layer for rating prediction (scale 1-5)
        outputs = layers.Dense(1, activation="sigmoid", name="rating_prediction")(x)
        outputs = layers.Lambda(lambda x: x * 4 + 1, name="rating_scale")(outputs)  # Scale to 1-5
        
        # Create model
        model = keras.Model(
            inputs=[user_inputs, movie_inputs],
            outputs=outputs,
            name="embedding_recommendation_model"
        )
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    @log_execution_time
    def fit(self, user_features, movie_features, ratings_df, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the embedding recommendation model.
        
        Parameters
        ----------
        user_features : pandas.DataFrame
            DataFrame containing user features.
        movie_features : pandas.DataFrame
            DataFrame containing movie features.
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
        logger.info("Training embedding recommendation model")
        
        # Prepare training data
        user_ids = ratings_df['userId'].values
        movie_ids = ratings_df['movieId'].values
        ratings = ratings_df['rating'].values
        
        # Get user features
        user_features_array = np.array([
            user_features.loc[user_features['userId'] == user_id].iloc[0].values[1:] 
            for user_id in user_ids
        ])
        
        # Get movie features
        movie_features_array = np.array([
            movie_features.loc[movie_features['movieId'] == movie_id].iloc[0].values[1:] 
            for movie_id in movie_ids
        ])
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_recommendation_model(
                user_features_array.shape[1],
                movie_features_array.shape[1]
            )
        
        # Train model
        history = self.model.fit(
            [user_features_array, movie_features_array],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self, history
    
    def get_embeddings(self, user_features=None, movie_features=None):
        """
        Get embeddings for users and/or movies.
        
        Parameters
        ----------
        user_features : pandas.DataFrame, optional
            DataFrame containing user features.
        movie_features : pandas.DataFrame, optional
            DataFrame containing movie features.
            
        Returns
        -------
        dict
            Dictionary containing user and/or movie embeddings.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        result = {}
        
        if user_features is not None:
            # Extract user IDs and features
            user_ids = user_features['userId'].values
            user_features_array = user_features.iloc[:, 1:].values
            
            # Get user embeddings
            user_embeddings, _ = self.user_embedding_model.predict(user_features_array)
            
            # Create DataFrame with user IDs and embeddings
            result['user_embeddings'] = {
                'ids': user_ids,
                'embeddings': user_embeddings
            }
        
        if movie_features is not None:
            # Extract movie IDs and features
            movie_ids = movie_features['movieId'].values
            movie_features_array = movie_features.iloc[:, 1:].values
            
            # Get movie embeddings
            movie_embeddings, _ = self.movie_embedding_model.predict(movie_features_array)
            
            # Create DataFrame with movie IDs and embeddings
            result['movie_embeddings'] = {
                'ids': movie_ids,
                'embeddings': movie_embeddings
            }
        
        return result
    
    @log_execution_time
    def predict(self, user_features, movie_features):
        """
        Predict ratings for user-movie pairs.
        
        Parameters
        ----------
        user_features : numpy.ndarray
            Array of user features.
        movie_features : numpy.ndarray
            Array of movie features.
            
        Returns
        -------
        numpy.ndarray
            Predicted ratings.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Predict ratings
        predictions = self.model.predict([user_features, movie_features])
        
        return predictions.flatten()
    
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
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the complete model
        model_path = os.path.join(model_dir, 'embedding_model.h5')
        self.model.save(model_path)
        
        # Save user embedding model
        user_model_path = os.path.join(model_dir, 'user_embedding_model.h5')
        self.user_embedding_model.save(user_model_path)
        
        # Save movie embedding model
        movie_model_path = os.path.join(model_dir, 'movie_embedding_model.h5')
        self.movie_embedding_model.save(movie_model_path)
        
        # Save configuration
        import json
        config_path = os.path.join(model_dir, 'embedding_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        
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
        EmbeddingRecommender
            The loaded model instance.
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(model_dir, 'embedding_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance with loaded configuration
        model = cls(config)
        
        # Load the complete model
        model_path = os.path.join(model_dir, 'embedding_model.h5')
        model.model = keras.models.load_model(model_path)
        
        # Load user embedding model
        user_model_path = os.path.join(model_dir, 'user_embedding_model.h5')
        model.user_embedding_model = keras.models.load_model(user_model_path)
        
        # Load movie embedding model
        movie_model_path = os.path.join(model_dir, 'movie_embedding_model.h5')
        model.movie_embedding_model = keras.models.load_model(movie_model_path)
        
        logger.info(f"Model loaded from {model_dir}")
        return model