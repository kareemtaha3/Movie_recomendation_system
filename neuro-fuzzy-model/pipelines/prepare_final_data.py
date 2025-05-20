import os
import sys
from pathlib import Path
import logging

# Add project root to sys.path to import from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import the merge_final_data function
from src.movie_recommender.data.merge_final_data import merge_final_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main(data_dir):
    """
    Execute the final data preparation pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory containing raw, interim, and processed folders
    """
    try:
        logger.info("Starting the final data preparation pipeline")
        
        # Ensure data_dir is an absolute path
        data_dir = Path(data_dir).resolve()
        logger.info(f"Using data directory: {data_dir}")
        
        # Step 1: Merge the final user data with movie metadata
        logger.info("Merging final user data with movie metadata")
        try:
            success, row_count = merge_final_data(data_dir=data_dir)
            if not success:
                raise Exception("Failed to merge data")
            logger.info(f"Successfully merged data with {row_count:,} rows")
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise
            
        logger.info("Final data preparation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the final data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    # Use default data directory path
    data_dir = project_root / "data"
    
    # Execute the main function
    main(data_dir=data_dir)