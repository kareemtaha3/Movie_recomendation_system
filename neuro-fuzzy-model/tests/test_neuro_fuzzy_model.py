import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from movie_recommender.models.neuro_fuzzy_model import NeuroFuzzyRecommender


class TestNeuroFuzzyModel(unittest.TestCase):
    """
    Test cases for the NeuroFuzzyRecommender class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a simple configuration
        self.config = {
            'neural_layers': [32, 16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu',
            'fuzzy_membership_funcs': 3,
            'embedding_dim': 16
        }
        
        # Create a model instance
        self.model = NeuroFuzzyRecommender(self.config)
        
        # Create sample data for testing
        self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample data for testing.
        """
        # Create sample movie features
        self.movie_features = pd.DataFrame({
            'movieId': range(1, 101),
            'year': np.random.randint(1980, 2020, 100),
            'year_norm': np.random.random(100),
            'rating_count': np.random.randint(10, 1000, 100),
            'rating_mean': np.random.uniform(1, 5, 100),
            'rating_std': np.random.uniform(0, 1, 100),
            'rating_min': np.random.uniform(1, 3, 100),
            'rating_max': np.random.uniform(3, 5, 100),
            'title': [f'Movie {i}' for i in range(1, 101)],
            'genres': ['Action|Adventure', 'Comedy|Romance', 'Drama|Thriller'] * 33 + ['Documentary']
        })
        
        # Add genre columns
        for genre in ['Action', 'Adventure', 'Comedy', 'Romance', 'Drama', 'Thriller', 'Documentary']:
            self.movie_features[f'genre_{genre}'] = np.random.randint(0, 2, 100)
        
        # Create sample user features
        self.user_features = pd.DataFrame({
            'userId': range(1, 51),
            'rating_count': np.random.randint(10, 200, 50),
            'rating_mean': np.random.uniform(1, 5, 50),
            'rating_std': np.random.uniform(0, 1, 50),
            'rating_min': np.random.uniform(1, 3, 50),
            'rating_max': np.random.uniform(3, 5, 50)
        })
        
        # Create sample ratings
        user_ids = np.random.choice(range(1, 51), 500)
        movie_ids = np.random.choice(range(1, 101), 500)
        ratings = np.random.uniform(1, 5, 500)
        
        self.ratings_df = pd.DataFrame({
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings
        })
    
    def test_initialization(self):
        """
        Test model initialization.
        """
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config['neural_layers'], [32, 16])
        self.assertEqual(self.model.config['dropout_rate'], 0.2)
        self.assertEqual(self.model.config['learning_rate'], 0.001)
        self.assertEqual(self.model.config['activation'], 'relu')
        self.assertEqual(self.model.config['fuzzy_membership_funcs'], 3)
        self.assertEqual(self.model.config['embedding_dim'], 16)
    
    def test_build_neural_network(self):
        """
        Test building the neural network component.
        """
        # Build a neural network with 10 input features
        neural_model = self.model._build_neural_network(10)
        
        # Check that the model was created successfully
        self.assertIsNotNone(neural_model)
        
        # Check the model architecture
        self.assertEqual(len(neural_model.layers), 6)  # Input, embedding, 2 hidden layers, 2 dropout layers, output
        
        # Check input shape
        self.assertEqual(neural_model.input_shape, (None, 10))
        
        # Check output shape
        self.assertEqual(neural_model.output_shape, (None, 1))
    
    def test_build_fuzzy_system(self):
        """
        Test building the fuzzy system component.
        """
        # Build a fuzzy system with two features
        feature_names = ['genre_match', 'year_match']
        fuzzy_system = self.model._build_fuzzy_system(feature_names)
        
        # Check that the fuzzy system was created successfully
        self.assertIsNotNone(fuzzy_system)
        
        # Check that the fuzzy variables were created
        self.assertIn('genre_match', fuzzy_system['variables'])
        self.assertIn('year_match', fuzzy_system['variables'])
        
        # Check that the output variable was created
        self.assertIsNotNone(fuzzy_system['output'])
        
        # Check that rules were created
        self.assertTrue(len(fuzzy_system['rules']) > 0)
    
    def test_fit_and_predict(self):
        """
        Test fitting the model and making predictions.
        """
        # Fit the model with a small number of epochs for testing
        self.model.fit(
            movie_features=self.movie_features,
            user_features=self.user_features,
            ratings_df=self.ratings_df,
            epochs=1,
            batch_size=32,
            validation_split=0.2
        )
        
        # Check that the model was trained successfully
        self.assertIsNotNone(self.model.neural_model)
        self.assertIsNotNone(self.model.fuzzy_system)
        
        # Make predictions for a few user-movie pairs
        user_ids = np.array([1, 2, 3])
        movie_ids = np.array([10, 20, 30])
        predictions = self.model.predict(user_ids, movie_ids)
        
        # Check that predictions were made successfully
        self.assertEqual(len(predictions), 3)
        
        # Check that predictions are within the valid rating range
        self.assertTrue(np.all(predictions >= 1.0))
        self.assertTrue(np.all(predictions <= 5.0))
    
    def test_recommend_movies(self):
        """
        Test movie recommendations for a user.
        """
        # Fit the model with a small number of epochs for testing
        self.model.fit(
            movie_features=self.movie_features,
            user_features=self.user_features,
            ratings_df=self.ratings_df,
            epochs=1,
            batch_size=32,
            validation_split=0.2
        )
        
        # Get recommendations for a user
        recommendations = self.model.recommend_movies(user_id=1, n=5)
        
        # Check that recommendations were made successfully
        self.assertEqual(len(recommendations), 5)
        
        # Check that recommendations include required columns
        self.assertIn('movieId', recommendations.columns)
        self.assertIn('predicted_rating', recommendations.columns)
        self.assertIn('title', recommendations.columns)
        
        # Check that recommendations are sorted by predicted rating in descending order
        self.assertTrue(recommendations['predicted_rating'].is_monotonic_decreasing)
    
    def test_save_and_load_model(self):
        """
        Test saving and loading the model.
        """
        # Fit the model with a small number of epochs for testing
        self.model.fit(
            movie_features=self.movie_features,
            user_features=self.user_features,
            ratings_df=self.ratings_df,
            epochs=1,
            batch_size=32,
            validation_split=0.2
        )
        
        # Create a temporary directory for saving the model
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the model
            model_dir = self.model.save_model(temp_dir)
            
            # Check that model files were created
            self.assertTrue(os.path.exists(os.path.join(model_dir, 'neural_model.h5')))
            self.assertTrue(os.path.exists(os.path.join(model_dir, 'fuzzy_system.joblib')))
            self.assertTrue(os.path.exists(os.path.join(model_dir, 'config.joblib')))
            
            # Load the model
            loaded_model = NeuroFuzzyRecommender.load_model(model_dir)
            
            # Check that the loaded model has the same configuration
            self.assertEqual(loaded_model.config, self.model.config)
            
            # Make predictions with both models
            user_ids = np.array([1, 2, 3])
            movie_ids = np.array([10, 20, 30])
            
            original_predictions = self.model.predict(user_ids, movie_ids)
            loaded_predictions = loaded_model.predict(user_ids, movie_ids)
            
            # Check that predictions are similar (may not be identical due to serialization/deserialization)
            np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()