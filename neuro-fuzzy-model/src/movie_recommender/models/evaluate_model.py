# Model evaluation script

import os
import logging
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
MODELS_DIR = PROJECT_DIR / 'artifacts' / 'models'
METRICS_DIR = PROJECT_DIR / 'artifacts' / 'metrics'
FIGURES_DIR = PROJECT_DIR / 'artifacts' / 'figures'
CONFIG_PATH = PROJECT_DIR / 'configs' / 'model_params.yaml'

def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_data():
    """Load the trained model, mappings, and test data."""
    logger.info("Loading model and data...")
    
    try:
        # Load model
        model_path = MODELS_DIR / 'final_model.h5'
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}. Run training first.")
            return None, None, None, None
        
        model = load_model(str(model_path))
        
        # Load mappings
        mappings_path = MODELS_DIR / 'mappings.joblib'
        if not mappings_path.exists():
            logger.error(f"Mappings file not found at {mappings_path}. Run training first.")
            return None, None, None, None
        
        mappings = joblib.load(mappings_path)
        
        # Load processed data
        user_item_matrix_path = PROCESSED_DATA_DIR / 'user_item_matrix.csv'
        if not user_item_matrix_path.exists():
            logger.error(f"User-item matrix not found at {user_item_matrix_path}. Run data preparation first.")
            return None, None, None, None
        
        user_item_df = pd.read_csv(user_item_matrix_path, index_col=0)
        
        # Load movies data
        movies_path = PROCESSED_DATA_DIR / 'processed_movies.csv'
        if not movies_path.exists():
            logger.error(f"Processed movies file not found at {movies_path}. Run data preparation first.")
            return None, None, None, None
        
        movies_df = pd.read_csv(movies_path)
        
        logger.info(f"Successfully loaded model and data")
        return model, mappings, user_item_df, movies_df
        
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        return None, None, None, None

def prepare_test_data(user_item_df, config):
    """Prepare test data for model evaluation."""
    if user_item_df is None:
        logger.error("Cannot prepare test data: missing input data")
        return None, None
    
    logger.info("Preparing test data...")
    
    try:
        # Convert to numpy array
        user_item_matrix = user_item_df.values
        
        # Create user and item indices
        user_indices = np.arange(user_item_matrix.shape[0])
        item_indices = np.arange(user_item_matrix.shape[1])
        
        # Create test data
        # For each non-zero entry in the matrix, create a test example
        user_idx, item_idx = user_item_matrix.nonzero()
        ratings = user_item_matrix[user_idx, item_idx]
        
        # Normalize ratings to [0, 1] range
        ratings_normalized = ratings / 5.0  # Assuming ratings are on a 1-5 scale
        
        # Create feature vectors
        X_test = np.column_stack((user_idx, item_idx))
        y_test = ratings_normalized
        
        logger.info(f"Prepared {len(X_test)} test samples")
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    if model is None or X_test is None or y_test is None:
        logger.error("Cannot evaluate model: missing model or test data")
        return None
    
    logger.info("Evaluating model...")
    
    try:
        # Prepare input data
        user_test = X_test[:, 0].astype(int)
        item_test = X_test[:, 1].astype(int)
        
        # Make predictions
        y_pred = model.predict([user_test, item_test], verbose=0)
        y_pred = y_pred.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate additional recommendation metrics
        # For simplicity, we'll use a threshold to convert ratings to binary relevance
        threshold = 0.7  # Equivalent to 3.5 on a 5-point scale
        y_test_binary = (y_test >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate precision, recall, and F1 score
        true_positives = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
        false_positives = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
        false_negatives = np.sum((y_test_binary == 1) & (y_pred_binary == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate accuracy
        accuracy = np.mean(y_test_binary == y_pred_binary)
        
        # Compile metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy)
        }
        
        logger.info(f"Model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}")
        return metrics, y_test, y_pred
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None, None, None

def generate_visualizations(y_test, y_pred, metrics):
    """Generate visualizations for model evaluation."""
    if y_test is None or y_pred is None or metrics is None:
        logger.error("Cannot generate visualizations: missing data")
        return False
    
    logger.info("Generating visualizations...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Scatter plot of actual vs predicted ratings
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Ratings (Normalized)')
        plt.ylabel('Predicted Ratings (Normalized)')
        plt.title('Actual vs Predicted Ratings')
        plt.savefig(FIGURES_DIR / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution of errors
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.savefig(FIGURES_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance metrics bar chart
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'accuracy']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_to_plot, y=[metrics[m] for m in metrics_to_plot])
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig(FIGURES_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Combined performance plot (for DVC tracking)
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5, s=20)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        # Subplot 2: Error Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(errors, kde=True)
        plt.xlabel('Error')
        plt.title('Error Distribution')
        
        # Subplot 3: Performance Metrics
        plt.subplot(2, 2, 3)
        sns.barplot(x=metrics_to_plot, y=[metrics[m] for m in metrics_to_plot])
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Performance Metrics')
        plt.ylim(0, 1)
        
        # Subplot 4: Error Metrics
        plt.subplot(2, 2, 4)
        error_metrics = ['mse', 'rmse', 'mae']
        sns.barplot(x=error_metrics, y=[metrics[m] for m in error_metrics])
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Error Metrics')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'performance_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {FIGURES_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return False

def save_evaluation_results(metrics):
    """Save evaluation metrics to a JSON file."""
    if metrics is None:
        logger.error("Cannot save evaluation results: missing metrics")
        return False
    
    logger.info("Saving evaluation results...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        # Save metrics
        metrics_path = METRICS_DIR / 'evaluation.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation results saved to {metrics_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        return False

def main():
    """Main function to run the model evaluation pipeline."""
    logger.info("Starting model evaluation pipeline")
    
    # Load configuration
    config = load_config()
    
    # Load model and data
    model, mappings, user_item_df, movies_df = load_model_and_data()
    
    if model is None or mappings is None:
        logger.error("Failed to load model or data. Exiting.")
        return
    
    # Prepare test data
    X_test, y_test = prepare_test_data(user_item_df, config)
    
    if X_test is None:
        logger.error("Failed to prepare test data. Exiting.")
        return
    
    # Evaluate model
    metrics, y_test, y_pred = evaluate_model(model, X_test, y_test)
    
    if metrics is None:
        logger.error("Failed to evaluate model. Exiting.")
        return
    
    # Generate visualizations
    viz_success = generate_visualizations(y_test, y_pred, metrics)
    
    # Save evaluation results
    save_success = save_evaluation_results(metrics)
    
    if viz_success and save_success:
        logger.info("Model evaluation pipeline completed successfully")
    else:
        logger.error("Model evaluation pipeline completed with errors")

if __name__ == "__main__":
    main()