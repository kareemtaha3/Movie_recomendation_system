#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training pipeline for the neuro-fuzzy movie recommendation system.

This script orchestrates the entire training pipeline, including:
1. Data preparation
2. Feature engineering
3. Model training
4. Model evaluation
5. Visualization of results

It can be run directly or imported as a module.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

# Add the src directory to the path so we can import our modules
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from movie_recommender.data.prepare_data import DataPreparation
from movie_recommender.features.feature_engineering import FeatureEngineering
from movie_recommender.models.neuro_fuzzy_model import NeuroFuzzyRecommender
from movie_recommender.visualization.visualize import RecommendationVisualizer
from movie_recommender.utils.logging import setup_logging, log_execution_time
from movie_recommender.utils.config import load_config, get_data_path, get_artifact_path


def parse_args():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train the neuro-fuzzy recommendation model')
    parser.add_argument('--config', type=str, default='configs/model_params.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation step')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


@log_execution_time
def prepare_data(config):
    """
    Prepare data for training.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
        
    Returns
    -------
    tuple
        Tuple containing (movies_df, ratings_df).
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info("Preparing data")
    
    # Initialize data preparation
    data_prep = DataPreparation(config)
    
    # Get data paths
    raw_data_dir = get_data_path(config, 'raw')
    processed_data_dir = get_data_path(config, 'processed')
    
    # Check if processed data already exists
    movies_processed_path = os.path.join(processed_data_dir, 'movies_processed.csv')
    ratings_processed_path = os.path.join(processed_data_dir, 'ratings_processed.csv')
    
    if os.path.exists(movies_processed_path) and os.path.exists(ratings_processed_path) and not config.get('force_reprocess', False):
        logger.info("Loading pre-processed data")
        movies_df = pd.read_csv(movies_processed_path)
        ratings_df = pd.read_csv(ratings_processed_path)
    else:
        logger.info("Processing raw data")
        # Load raw data
        movies_df, ratings_df = data_prep.load_data(raw_data_dir)
        
        # Preprocess data
        movies_df, ratings_df = data_prep.preprocess_data(movies_df, ratings_df)
        
        # Save processed data
        os.makedirs(processed_data_dir, exist_ok=True)
        movies_df.to_csv(movies_processed_path, index=False)
        ratings_df.to_csv(ratings_processed_path, index=False)
        logger.info(f"Saved processed data to {processed_data_dir}")
    
    return movies_df, ratings_df


@log_execution_time
def engineer_features(config, movies_df, ratings_df):
    """
    Engineer features for the model.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
    movies_df : pandas.DataFrame
        DataFrame containing movie information.
    ratings_df : pandas.DataFrame
        DataFrame containing user ratings.
        
    Returns
    -------
    tuple
        Tuple containing (movie_features, user_features, ratings_df).
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info("Engineering features")
    
    # Initialize feature engineering
    feature_eng = FeatureEngineering(config.get('feature_engineering', {}))
    
    # Engineer features
    movie_features, user_features, ratings_df = feature_eng.prepare_features_for_model(
        movies_df, ratings_df
    )
    
    return movie_features, user_features, ratings_df


@log_execution_time
def train_model(config, movie_features, user_features, ratings_df):
    """
    Train the neuro-fuzzy recommendation model.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
    movie_features : pandas.DataFrame
        DataFrame containing movie features.
    user_features : pandas.DataFrame
        DataFrame containing user features.
    ratings_df : pandas.DataFrame
        DataFrame containing user ratings.
        
    Returns
    -------
    NeuroFuzzyRecommender
        Trained model.
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info("Training model")
    
    # Initialize model
    model = NeuroFuzzyRecommender(config.get('model', {}))
    
    # Train model
    model.fit(
        movie_features=movie_features,
        user_features=user_features,
        ratings_df=ratings_df,
        epochs=config.get('training', {}).get('epochs', 10),
        batch_size=config.get('training', {}).get('batch_size', 32),
        validation_split=config.get('training', {}).get('validation_split', 0.2)
    )
    
    # Save model
    model_dir = get_artifact_path(config, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'neuro_fuzzy_model')
    model.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    return model


@log_execution_time
def evaluate_model(config, model, movie_features, user_features, ratings_df):
    """
    Evaluate the trained model.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
    model : NeuroFuzzyRecommender
        Trained model.
    movie_features : pandas.DataFrame
        DataFrame containing movie features.
    user_features : pandas.DataFrame
        DataFrame containing user features.
    ratings_df : pandas.DataFrame
        DataFrame containing user ratings.
        
    Returns
    -------
    dict
        Evaluation metrics.
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info("Evaluating model")
    
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
    mse = np.mean((predicted_ratings - actual_ratings) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_ratings - actual_ratings))
    
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
    
    # Save metrics
    metrics_dir = get_artifact_path(config, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'evaluation_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    # Save test predictions for further analysis
    predictions_df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'actual_rating': actual_ratings,
        'predicted_rating': predicted_ratings
    })
    predictions_path = os.path.join(metrics_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved test predictions to {predictions_path}")
    
    return metrics, predictions_df


@log_execution_time
def generate_visualizations(config, metrics, predictions_df, movie_features, user_features, ratings_df):
    """
    Generate visualizations for model evaluation.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
    metrics : dict
        Evaluation metrics.
    predictions_df : pandas.DataFrame
        DataFrame containing test predictions.
    movie_features : pandas.DataFrame
        DataFrame containing movie features.
    user_features : pandas.DataFrame
        DataFrame containing user features.
    ratings_df : pandas.DataFrame
        DataFrame containing user ratings.
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info("Generating visualizations")
    
    # Initialize visualizer
    figures_dir = get_artifact_path(config, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    visualizer = RecommendationVisualizer(output_dir=figures_dir)
    
    # Plot rating distribution
    visualizer.plot_rating_distribution(
        ratings_df,
        title="Rating Distribution",
        save_as="rating_distribution.png"
    )
    
    # Plot predicted vs actual ratings
    visualizer.plot_prediction_vs_actual(
        predictions_df['actual_rating'],
        predictions_df['predicted_rating'],
        title="Predicted vs Actual Ratings",
        save_as="predicted_vs_actual.png"
    )
    
    # Plot error distribution
    visualizer.plot_error_distribution(
        predictions_df['actual_rating'],
        predictions_df['predicted_rating'],
        title="Error Distribution",
        save_as="error_distribution.png"
    )
    
    # Plot recommendation metrics
    visualizer.plot_recommendation_metrics(
        metrics,
        title="Recommendation Metrics",
        save_as="recommendation_metrics.png"
    )
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        predictions_df['actual_rating'],
        predictions_df['predicted_rating'],
        threshold=config.get('evaluation', {}).get('like_threshold', 3.5),
        title="Confusion Matrix",
        save_as="confusion_matrix.png"
    )
    
    # Plot precision-recall curve
    visualizer.plot_precision_recall_curve(
        predictions_df['actual_rating'],
        predictions_df['predicted_rating'],
        threshold=config.get('evaluation', {}).get('like_threshold', 3.5),
        title="Precision-Recall Curve",
        save_as="precision_recall_curve.png"
    )
    
    # Plot ROC curve
    visualizer.plot_roc_curve(
        predictions_df['actual_rating'],
        predictions_df['predicted_rating'],
        threshold=config.get('evaluation', {}).get('like_threshold', 3.5),
        title="ROC Curve",
        save_as="roc_curve.png"
    )
    
    # Plot genre distribution
    visualizer.plot_genre_distribution(
        movie_features,
        title="Genre Distribution",
        save_as="genre_distribution.png"
    )
    
    # Plot user activity
    visualizer.plot_user_activity(
        ratings_df,
        title="User Activity Distribution",
        save_as="user_activity.png"
    )
    
    logger.info(f"Saved visualizations to {figures_dir}")


def main():
    """
    Main function to run the training pipeline.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file=log_file, log_level=getattr(logging, args.log_level))
    
    logger.info("Starting training pipeline")
    logger.info(f"Configuration: {config}")
    
    try:
        # Prepare data
        movies_df, ratings_df = prepare_data(config)
        
        # Engineer features
        movie_features, user_features, ratings_df = engineer_features(config, movies_df, ratings_df)
        
        # Train model
        model = train_model(config, movie_features, user_features, ratings_df)
        
        # Evaluate model
        if not args.no_eval:
            metrics, predictions_df = evaluate_model(config, model, movie_features, user_features, ratings_df)
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Generate visualizations
            if not args.no_plots:
                generate_visualizations(config, metrics, predictions_df, movie_features, user_features, ratings_df)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in training pipeline: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()