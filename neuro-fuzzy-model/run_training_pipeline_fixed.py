#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the training pipeline for the neuro-fuzzy movie recommendation system.

This script sets up the environment and executes the training pipeline with proper
error handling and logging.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the training pipeline
from pipelines.train_pipeline_fixed import main as run_pipeline
from src.movie_recommender.utils.logging import setup_logging


def parse_args():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the neuro-fuzzy recommendation model training pipeline')
    parser.add_argument('--config', type=str, default='configs/model_params.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--features-dir', type=str, default=None,
                        help='Directory containing processed features')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save trained models and artifacts')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation step')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


def setup_directories():
    """
    Set up necessary directories for the training pipeline.
    
    Returns
    -------
    dict
        Dictionary containing paths to the created directories.
    """
    # Create necessary directories if they don't exist
    directories = {
        'data': os.path.join(project_root, 'data'),
        'raw': os.path.join(project_root, 'data', 'raw'),
        'processed': os.path.join(project_root, 'data', 'processed'),
        'artifacts': os.path.join(project_root, 'artifacts'),
        'models': os.path.join(project_root, 'artifacts', 'models'),
        'metrics': os.path.join(project_root, 'artifacts', 'metrics'),
        'figures': os.path.join(project_root, 'artifacts', 'figures'),
        'logs': os.path.join(project_root, 'logs')
    }
    
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories


def main():
    """
    Main function to run the training pipeline.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    directories = setup_directories()
    
    # Setup logging
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(directories['logs'], f"training_run_{timestamp}.log")
    logger = setup_logging(log_file=log_file, log_level=getattr(logging, args.log_level))
    
    logger.info("Starting neuro-fuzzy recommendation model training")
    logger.info(f"Arguments: {args}")
    
    try:
        # Run the training pipeline
        run_pipeline(
            features_dir=args.features_dir,
            output_dir=args.output_dir
        )
        
        logger.info("Training completed successfully")
        print("\nTraining completed successfully!")
        print(f"Model artifacts saved to: {os.path.abspath(directories['artifacts'])}")
        print(f"Log file: {os.path.abspath(log_file)}")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        print(f"\nError during training: {e}")
        print(f"See log file for details: {os.path.abspath(log_file)}")
        sys.exit(1)


if __name__ == '__main__':
    main()