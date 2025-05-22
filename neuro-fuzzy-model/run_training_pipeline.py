"""Script to run the training pipeline using the processed features from feature engineering."""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to path to allow imports
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

from src.movie_recommender.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main(features_dir, output_dir, model_type='neuro_fuzzy'):
    """Run the training pipeline with the specified parameters.
    
    Args:
        features_dir: Directory containing processed features
        output_dir: Directory to save trained models
        model_type: Type of model to train ('neuro_fuzzy' or 'embedding_neuro_fuzzy')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if features directory exists
    if not os.path.exists(features_dir):
        logger.error(f"Features directory not found at {features_dir}")
        print(f"\nERROR: Features directory not found. Please run the feature engineering pipeline first.")
        return
    
    # Check if required feature files exist
    required_files = ['movie_features.parquet', 'user_profiles.parquet', 'interaction_features.parquet']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(features_dir, f))]
    
    if missing_files:
        logger.error(f"Missing required feature files: {missing_files}")
        print(f"\nERROR: Missing required feature files in {features_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please run the feature engineering pipeline first.")
        return
    
    # Import the appropriate training pipeline based on model type
    if model_type == 'neuro_fuzzy':
        from pipelines.train_pipeline import main as run_training
        model_name = "Neuro-Fuzzy"
    elif model_type == 'embedding_neuro_fuzzy':
        from pipelines.train_embedding_neuro_fuzzy_pipeline import main as run_training
        model_name = "Embedding Neuro-Fuzzy"
    else:
        logger.error(f"Unknown model type: {model_type}")
        print(f"\nERROR: Unknown model type: {model_type}")
        print("Supported model types: 'neuro_fuzzy', 'embedding_neuro_fuzzy'")
        return
    
    # Run the training pipeline
    print(f"\nStarting {model_name} training pipeline...")
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Run the training pipeline
    run_training(
        features_dir=features_dir,
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Trained model saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--features_dir', type=str, 
                      default=os.path.join(project_dir, 'data', 'processed'),
                      help='Directory containing processed features')
    parser.add_argument('--output_dir', type=str, 
                     default=os.path.join(project_dir, 'models'),
                     help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, 
                     default='neuro_fuzzy',
                     choices=['neuro_fuzzy', 'embedding_neuro_fuzzy'],
                     help='Type of model to train')
    
    args = parser.parse_args()
    
    main(args.features_dir, args.output_dir, args.model_type)