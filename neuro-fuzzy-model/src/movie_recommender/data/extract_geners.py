import pandas as pd
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'raw'

def extract_genres():
    """
    Extract unique genres from the movies dataset, assign IDs, and save to a new file.
    Returns a dictionary mapping genre names to their IDs.
    """
    try:
        # Load the movies data
        movies_path = RAW_DATA_DIR / 'movies.csv'
        if not movies_path.exists():
            logger.error(f"Movies data file not found at {movies_path}")
            return None

        logger.info("Loading movies data...")
        movies_df = pd.read_csv(movies_path)

        # Extract all unique genres
        logger.info("Extracting unique genres...")
        all_genres = set()
        for genres_str in movies_df['genres']:
            if pd.notna(genres_str):  # Handle NaN values
                genres = genres_str.split('|')
                all_genres.update(genres)

        # Remove 'no genres listed' if it exists
        all_genres.discard('(no genres listed)')

        # Create genre mapping
        genre_mapping = {genre: idx + 1 for idx, genre in enumerate(sorted(all_genres))}

        # Create a DataFrame for genres
        genres_df = pd.DataFrame([
            {'genre_id': genre_id, 'genre_name': genre_name}
            for genre_name, genre_id in genre_mapping.items()
        ])

        # Save genres to CSV
        output_path = PROCESSED_DATA_DIR / 'genres.csv'
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        genres_df.to_csv(output_path, index=False)

        logger.info(f"Successfully extracted {len(genre_mapping)} unique genres")
        logger.info(f"Genres data saved to {output_path}")

        return genre_mapping

    except Exception as e:
        logger.error(f"Error extracting genres: {e}")
        return None

def main():
    """Main function to run the genre extraction process."""
    logger.info("Starting genre extraction process")
    genre_mapping = extract_genres()
    
    if genre_mapping:
        logger.info("Genre extraction completed successfully")
        # Print some statistics
        logger.info(f"Total number of unique genres: {len(genre_mapping)}")
        logger.info("First few genres:")
        for genre, genre_id in list(genre_mapping.items())[:5]:
            logger.info(f"Genre ID {genre_id}: {genre}")
    else:
        logger.error("Genre extraction failed")

if __name__ == "__main__":
    main()
