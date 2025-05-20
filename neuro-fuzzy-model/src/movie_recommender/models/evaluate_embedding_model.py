# Embedding model evaluation script

import os
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split

from movie_recommender.models.embedding_neuro_fuzzy_model import EmbeddingNeuroFuzzyRecommender
from movie_recommender.utils.logging import get_logger, log_execution_time

# Setup logging
logger = get_logger(__name__)

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
CONFIG_PATH = BASE_DIR / 'configs' / 'embedding_model_params.yaml'
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
METRICS_DIR = BASE_DIR / 'metrics'
FIGURES_DIR = BASE_DIR / 'figures'


def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_data():
    """Load the trained model and data."""
    logger.info("Loading model and data...")
    
    try:
        # Load model
        model_path = MODELS_DIR / 'embedding_neuro_fuzzy_model'
        if not model_path.exists():
            logger.error(f"Model directory not found at {model_path}. Run training first.")
            return None, None, None, None
        
        model = EmbeddingNeuroFuzzyRecommender.load_model(str(model_path))
        
        # Load processed data
        movies_processed_path = PROCESSED_DATA_DIR / 'movies_processed.csv'
        ratings_processed_path = PROCESSED_DATA_DIR / 'ratings_processed.csv'
        
        if not movies_processed_path.exists() or not ratings_processed_path.exists():
            logger.error(f"Processed data files not found. Run data preparation first.")
            return None, None, None, None
        
        movies_df = pd.read_csv(movies_processed_path)
        ratings_df = pd.read_csv(ratings_processed_path)
        
        # Load feature data
        movie_features_path = PROCESSED_DATA_DIR / 'movie_features.csv'
        user_features_path = PROCESSED_DATA_DIR / 'user_features.csv'
        
        if not movie_features_path.exists() or not user_features_path.exists():
            logger.error(f"Feature data files not found. Run feature engineering first.")
            return None, None, None, None
        
        movie_features = pd.read_csv(movie_features_path)
        user_features = pd.read_csv(user_features_path)
        
        logger.info(f"Loaded model and data successfully")
        return model, movie_features, user_features, ratings_df
        
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        return None, None, None, None


@log_execution_time
def evaluate_model(model, movie_features, user_features, ratings_df, config):
    """Evaluate the model on test data."""
    if model is None or movie_features is None or user_features is None or ratings_df is None:
        logger.error("Cannot evaluate model: missing model or data")
        return None, None
    
    logger.info("Evaluating embedding neuro-fuzzy model...")
    
    try:
        # Split data into train and test sets
        test_size = config.get('evaluation', {}).get('test_size', 0.2)
        random_state = config.get('evaluation', {}).get('random_seed', 42)
        
        # Shuffle and split the ratings data
        np.random.seed(random_state)
        indices = np.random.permutation(len(ratings_df))
        test_size_int = int(test_size * len(ratings_df))
        test_indices = indices[:test_size_int]
        
        test_ratings = ratings_df.iloc[test_indices]
        
        # Prepare test data
        user_ids = test_ratings['userId'].values
        movie_ids = test_ratings['movieId'].values
        actual_ratings = test_ratings['rating'].values
        
        # Make predictions
        predicted_ratings = model.predict(user_ids, movie_ids)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        
        # Calculate additional metrics
        correlation = np.corrcoef(actual_ratings, predicted_ratings)[0, 1]
        
        # Calculate binary classification metrics
        threshold = config.get('evaluation', {}).get('like_threshold', 3.5)
        actual_binary = (actual_ratings >= threshold).astype(int)
        predicted_binary = (predicted_ratings >= threshold).astype(int)
        
        # Calculate accuracy, precision, recall, and F1 score
        accuracy = np.mean(actual_binary == predicted_binary)
        precision = np.sum((actual_binary == 1) & (predicted_binary == 1)) / (np.sum(predicted_binary == 1) + 1e-10)
        recall = np.sum((actual_binary == 1) & (predicted_binary == 1)) / (np.sum(actual_binary == 1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Compile metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'userId': user_ids,
            'movieId': movie_ids,
            'actual_rating': actual_ratings,
            'predicted_rating': predicted_ratings
        })
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics, predictions_df
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None, None


@log_execution_time
def generate_recommendations(model, movie_features, user_features, config, n_users=5, n_recommendations=10):
    """Generate recommendations for a sample of users."""
    if model is None or movie_features is None or user_features is None:
        logger.error("Cannot generate recommendations: missing model or data")
        return None
    
    logger.info(f"Generating recommendations for {n_users} sample users...")
    
    try:
        # Select random users
        sample_users = np.random.choice(user_features['userId'].unique(), 
                                       min(n_users, len(user_features['userId'].unique())), 
                                       replace=False)
        
        recommendations = {}
        
        for user_id in sample_users:
            # Get recommendations for this user
            user_recommendations = model.recommend_movies(user_id, n=n_recommendations)
            recommendations[user_id] = user_recommendations
            
            logger.info(f"Generated {len(user_recommendations)} recommendations for user {user_id}")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return None


@log_execution_time
def plot_evaluation_results(metrics, predictions_df, config):
    """Plot evaluation results."""
    if metrics is None or predictions_df is None:
        logger.error("Cannot plot evaluation results: missing metrics or predictions")
        return
    
    logger.info("Plotting evaluation results...")
    
    try:
        # Create figures directory if it doesn't exist
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Plot predicted vs actual ratings
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions_df['actual_rating'], predictions_df['predicted_rating'], alpha=0.1)
        plt.plot([1, 5], [1, 5], 'r--')
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        plt.title('Predicted vs Actual Ratings (Embedding Neuro-Fuzzy Model)')
        plt.savefig(os.path.join(FIGURES_DIR, 'embedding_predicted_vs_actual.png'))
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        errors = predictions_df['predicted_rating'] - predictions_df['actual_rating']
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution (Embedding Neuro-Fuzzy Model)')
        plt.savefig(os.path.join(FIGURES_DIR, 'embedding_error_distribution.png'))
        plt.close()
        
        # Plot metrics as a bar chart
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['rmse', 'mae', 'correlation', 'accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics[m] for m in metrics_to_plot]
        plt.bar(metrics_to_plot, values)
        plt.ylabel('Value')
        plt.title('Evaluation Metrics (Embedding Neuro-Fuzzy Model)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'embedding_evaluation_metrics.png'))
        plt.close()
        
        # Plot confusion matrix
        threshold = config.get('evaluation', {}).get('like_threshold', 3.5)
        actual_binary = (predictions_df['actual_rating'] >= threshold).astype(int)
        predicted_binary = (predictions_df['predicted_rating'] >= threshold).astype(int)
        
        conf_matrix = pd.crosstab(actual_binary, predicted_binary, 
                                rownames=['Actual'], colnames=['Predicted'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Embedding Neuro-Fuzzy Model)')
        plt.savefig(os.path.join(FIGURES_DIR, 'embedding_confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(actual_binary, predictions_df['predicted_rating'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Embedding Neuro-Fuzzy Model)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(FIGURES_DIR, 'embedding_roc_curve.png'))
        plt.close()
        
        logger.info(f"Saved evaluation plots to {FIGURES_DIR}")
        
    except Exception as e:
        logger.error(f"Error plotting evaluation results: {e}")


@log_execution_time
def save_results(metrics, predictions_df, recommendations=None):
    """Save evaluation results and recommendations."""
    if metrics is None or predictions_df is None:
        logger.error("Cannot save results: missing metrics or predictions")
        return
    
    logger.info("Saving evaluation results...")
    
    try:
        # Create metrics directory if it doesn't exist
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(METRICS_DIR, 'embedding_evaluation_metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f)
        
        # Save predictions
        predictions_path = os.path.join(METRICS_DIR, 'embedding_test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save recommendations if available
        if recommendations is not None:
            recommendations_dir = os.path.join(METRICS_DIR, 'embedding_recommendations')
            os.makedirs(recommendations_dir, exist_ok=True)
            
            for user_id, user_recommendations in recommendations.items():
                recommendations_path = os.path.join(recommendations_dir, f'user_{user_id}_recommendations.csv')
                user_recommendations.to_csv(recommendations_path, index=False)
        
        logger.info(f"Saved evaluation results to {METRICS_DIR}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    """Main function to run the model evaluation pipeline."""
    logger.info("Starting embedding neuro-fuzzy model evaluation pipeline")
    
    # Load configuration
    config = load_config()
    
    # Load model and data
    model, movie_features, user_features, ratings_df = load_model_and_data()
    
    if model is None or movie_features is None or user_features is None or ratings_df is None:
        logger.error("Failed to load model or data. Exiting.")
        return
    
    # Evaluate model
    metrics, predictions_df = evaluate_model(model, movie_features, user_features, ratings_df, config)
    
    if metrics is None or predictions_df is None:
        logger.error("Failed to evaluate model. Exiting.")
        return
    
    # Generate recommendations
    recommendations = generate_recommendations(model, movie_features, user_features, config)
    
    # Plot evaluation results
    plot_evaluation_results(metrics, predictions_df, config)
    
    # Save results
    save_results(metrics, predictions_df, recommendations)
    
    logger.info("Embedding neuro-fuzzy model evaluation pipeline completed successfully")


if __name__ == '__main__':
    main()