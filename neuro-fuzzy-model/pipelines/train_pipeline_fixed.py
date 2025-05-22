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

# from movie_recommender.data.prepare_data import DataPreparation
from movie_recommender.features.feature_engineering import FeatureEngineering
from movie_recommender.models.neuro_fuzzy_model_fixed import NeuroFuzzyRecommender
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
    parser.add_argument('--features-dir', type=str, default=None,
                        help='Directory containing processed features')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save trained models and artifacts')
    return parser.parse_args()


@log_execution_time
def load_features(features_dir):
    """
    Load pre-processed features from the specified directory.
    
    Parameters
    ----------
    features_dir : str
        Directory containing processed features.
        
    Returns
    -------
    tuple
        Tuple containing (movie_features, user_features, ratings_df).
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info(f"Loading features from {features_dir}")
    
    # Check if required files exist
    movie_features_path = os.path.join(features_dir, 'movie_features.parquet')
    user_features_path = os.path.join(features_dir, 'user_profiles.parquet')
    interaction_features_path = os.path.join(features_dir, 'interaction_features.parquet')
    
    if not all(os.path.exists(p) for p in [movie_features_path, user_features_path, interaction_features_path]):
        missing = [p for p in [movie_features_path, user_features_path, interaction_features_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing required feature files: {missing}")
    
    # Load features
    movie_features = pd.read_parquet(movie_features_path)
    user_features = pd.read_parquet(user_features_path)
    ratings_df = pd.read_parquet(interaction_features_path)
    
    logger.info(f"Loaded {len(movie_features)} movie features, {len(user_features)} user features, and {len(ratings_df)} ratings")
    return movie_features, user_features, ratings_df


@log_execution_time
def load_merged_data(features_dir):
    """
    Load and process the final_merged_data.parquet file to extract movie features, user features, and ratings.
    Uses row group-based reading to reduce memory usage.
    
    Parameters
    ----------
    features_dir : str
        Directory containing the final_merged_data.parquet file.
        
    Returns
    -------
    tuple
        Tuple containing (movie_features, user_features, ratings_df).
    """
    logger = logging.getLogger('movie_recommender.pipeline')
    logger.info(f"Loading merged data from {features_dir}")
    
    # Check if the merged data file exists
    merged_data_path = os.path.join(features_dir, 'final_merged_data.parquet')
    
    if not os.path.exists(merged_data_path):
        raise FileNotFoundError(f"Merged data file not found: {merged_data_path}")
    
    # Define columns we need
    movie_columns = [
        'movieId', 'title', 'genres', 'imdbId', 'tmdbId',
        'budget', 'original_language', 'popularity', 'release_date', 
        'revenue', 'runtime', 'vote_average', 'vote_count',
        'cast', 'crew', 'production_companies'
    ]
    
    ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
    
    # Import pyarrow modules
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    # Open the parquet file
    parquet_file = pq.ParquetFile(merged_data_path)
    file_schema = parquet_file.schema.names
    
    # Filter to only columns that exist in the file
    available_movie_columns = [col for col in movie_columns if col in file_schema]
    available_ratings_columns = [col for col in ratings_columns if col in file_schema]
    
    # Combine all needed columns
    available_columns = list(set(available_movie_columns + available_ratings_columns))
    
    logger.info(f"Reading only necessary columns: {available_columns}")
    
    # Process in chunks by row groups
    num_row_groups = parquet_file.num_row_groups
    logger.info(f"Processing {num_row_groups} row groups")
    
    # Initialize empty DataFrames for collecting data
    all_movie_data = []
    all_user_ids = set()
    all_ratings_data = []
    
    # Process each row group separately
    for i in range(num_row_groups):
        logger.info(f"Processing row group {i+1}/{num_row_groups}")
        
        # Read only the columns we need from this row group
        table = parquet_file.read_row_group(i, columns=available_columns)
        chunk_df = table.to_pandas()
        
        # Extract movie features from this chunk
        if i == 0:  # Only extract movie features from the first chunk to save memory
            chunk_movie_features = chunk_df[available_movie_columns].drop_duplicates(subset=['movieId'])
            all_movie_data.append(chunk_movie_features)
        
        # Collect unique user IDs
        all_user_ids.update(chunk_df['userId'].unique())
        
        # Extract ratings
        chunk_ratings = chunk_df[available_ratings_columns]
        all_ratings_data.append(chunk_ratings)
        
        # Clear memory
        del chunk_df, table
        import gc
        gc.collect()
    
    # Combine the collected data
    movie_features = pd.concat(all_movie_data, ignore_index=True).drop_duplicates(subset=['movieId'])
    user_features = pd.DataFrame({'userId': list(all_user_ids)})
    ratings_df = pd.concat(all_ratings_data, ignore_index=True)
    
    logger.info(f"Extracted {len(movie_features)} movie features, {len(user_features)} user features, and {len(ratings_df)} ratings")
    return movie_features, user_features, ratings_df


def main(features_dir=None, output_dir=None):
    """
    Main function to run the training pipeline.
    
    Parameters
    ----------
    features_dir : str, optional
        Directory containing processed features.
    output_dir : str, optional
        Directory to save trained models and artifacts.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Override with function arguments if provided
    if features_dir is not None:
        args.features_dir = features_dir
    if output_dir is not None:
        args.output_dir = output_dir
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file=log_file, log_level=getattr(logging, args.log_level))
    
    logger.info("Starting training pipeline")
    logger.info(f"Configuration: {config}")
    
    # Set output directory if provided
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    try:
        # Load features
        if args.features_dir:
            features_dir = args.features_dir
        else:
            # Use default data paths from config
            features_dir = get_data_path(config, 'processed')
        
        # Try to load the merged data first, if that fails, fall back to separate feature files
        try:
            logger.info("Attempting to load merged data...")
            movie_features, user_features, ratings_df = load_merged_data(features_dir)
        except FileNotFoundError as e:
            logger.warning(f"Could not load merged data: {e}")
            logger.info("Falling back to separate feature files...")
            movie_features, user_features, ratings_df = load_features(features_dir)
        
        # Train model
        model = train_model(config, movie_features, user_features, ratings_df)
        
        # Evaluate model
        if not args.no_eval:
            metrics, predictions_df = evaluate_model(config, model, movie_features, user_features, ratings_df)
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Generate visualizations
            if not args.no_plots:
                generate_visualizations(config, model, metrics, predictions_df, movie_features, user_features, ratings_df)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in training pipeline: {e}")
        sys.exit(1)


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
    
    # Get model parameters from config
    model_params = config.get('model', {})
    training_params = config.get('training', {})
    
    # Initialize model
    model = NeuroFuzzyRecommender(model_params)
    
    # Train model
    model.fit(
        movie_features=movie_features,
        user_features=user_features,
        ratings_df=ratings_df,
        epochs=training_params.get('epochs', 100),
        batch_size=training_params.get('batch_size', 64),
        validation_split=training_params.get('validation_split', 0.2)
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
        'mse': float(mse),  # Convert numpy types to Python types for YAML serialization
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
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
def generate_visualizations(config, model, metrics, predictions_df, movie_features, user_features, ratings_df):
    """
    Generate visualizations for model evaluation.
    
    Parameters
    ----------
    config : dict
        Configuration parameters.
    model : NeuroFuzzyRecommender
        Trained model.
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
    
    # Plot training history
    try:
        history_fig = model.plot_training_history()
        history_path = os.path.join(figures_dir, 'training_history.png')
        history_fig.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close(history_fig)
        logger.info(f"Saved training history plot to {history_path}")
    except Exception as e:
        logger.warning(f"Could not generate training history plot: {e}")
    
    # Plot fuzzy membership functions
    try:
        fuzzy_fig = model.plot_fuzzy_membership()
        fuzzy_path = os.path.join(figures_dir, 'fuzzy_membership.png')
        fuzzy_fig.savefig(fuzzy_path, dpi=300, bbox_inches='tight')
        plt.close(fuzzy_fig)
        logger.info(f"Saved fuzzy membership plot to {fuzzy_path}")
    except Exception as e:
        logger.warning(f"Could not generate fuzzy membership plot: {e}")
    
    logger.info(f"Saved visualizations to {figures_dir}")


if __name__ == '__main__':
    main()