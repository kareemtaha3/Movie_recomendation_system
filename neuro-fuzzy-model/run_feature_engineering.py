"""Script to run the feature engineering pipeline with optimized parallel processing."""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path to allow imports
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

from pipelines.apply_feature_engineering import main as run_feature_engineering
from src.movie_recommender.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Run the feature engineering pipeline with default parameters."""
    # Define input and output paths
    input_filepath = os.path.join(project_dir, 'data', 'interim', 'final_merged_data.parquet')
    output_dir = os.path.join(project_dir, 'data', 'processed')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(input_filepath), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_filepath):
        # For testing, we'll use a CSV file if the parquet doesn't exist
        input_filepath = os.path.join(project_dir, 'data', 'interim', 'final_merged_data.csv')
        if not os.path.exists(input_filepath):
            logger.error(f"Input file not found at {input_filepath}")
            print(f"\nERROR: Input file not found. Please make sure your data file exists at:")
            print(f"  - {os.path.join(project_dir, 'data', 'interim', 'final_merged_data.parquet')}")
            print(f"  - or {os.path.join(project_dir, 'data', 'interim', 'final_merged_data.csv')}")
            return
    
    # Run the feature engineering pipeline
    print(f"\nStarting feature engineering pipeline...")
    print(f"Input file: {input_filepath}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Run with default parameters
    run_feature_engineering(
        input_filepath=input_filepath,
        output_dir=output_dir,
        chunk_size=50000,  # Default chunk size
        n_workers=None     # Use optimal number of workers
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nFeature engineering completed in {elapsed_time:.2f} seconds")
    print(f"Processed features saved to: {output_dir}")

if __name__ == "__main__":
    main()