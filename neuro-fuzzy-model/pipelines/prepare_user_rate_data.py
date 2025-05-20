import logging
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
project_root = Path(__file__).resolve().parent.parent
src_path = str(project_root / "src")
sys.path.append(src_path)

from movie_recommender.data import merge_ratings_movies, merge_with_links

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def prepare_user_rating_data():
    """
    Pipeline to prepare user rating data by merging ratings, movies, and links data.
    """
    try:
        # Create necessary directories
        data_dir = project_root / "data"
        raw_dir = data_dir / "raw"
        merged_dir = data_dir / "merged"
        
        # Define file paths
        ratings_path = str(raw_dir / "ratings.csv")
        movies_path = str(raw_dir / "movies.csv")
        links_path = str(raw_dir / "links.csv")
        
        # Intermediate and final output paths
        intermediate_parquet = str(merged_dir / "merged_ratings_movies.parquet")
        final_parquet = str(merged_dir / "final_user_data.parquet")
        
        # Log the start of the pipeline
        logger.info("Starting user rating data preparation pipeline")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Data directory: {data_dir}")
        
        # Step 1: Merge ratings and movies
        logger.info("Step 1: Merging ratings and movies data")
        columns_to_keep = [
            'userId',
            'movieId',
            'rating',
            'title',
            'genres'
        ]
        
        success, message = merge_ratings_movies(
            ratings_path=ratings_path,
            movies_path=movies_path,
            output_path=intermediate_parquet,
            chunk_size=500000,
            columns_to_keep=columns_to_keep
        )
        
        if not success:
            raise Exception(f"Failed to merge ratings and movies: {message}")
        
        logger.info("Successfully merged ratings and movies data")
        
        # Step 2: Merge with links data
        logger.info("Step 2: Merging with links data")
        columns_to_keep.extend(['imdbId', 'tmdbId'])
        
        success, message = merge_with_links(
            parquet_path=intermediate_parquet,
            links_path=links_path,
            output_path=final_parquet,
            chunk_size=500000,
            columns_to_keep=columns_to_keep
        )
        
        if not success:
            raise Exception(f"Failed to merge with links: {message}")
        
        logger.info("Successfully merged all data")
        
        # Clean up intermediate file
        if os.path.exists(intermediate_parquet):
            os.remove(intermediate_parquet)
            logger.info("Cleaned up intermediate file")
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Final merged data saved to: {final_parquet}")
        
        return True, "Data preparation completed successfully"
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

if __name__ == "__main__":
    success, message = prepare_user_rating_data()
    if not success:
        sys.exit(1)
