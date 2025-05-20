import sys
from pathlib import Path  # Added missing import
import pandas as pd
from pathlib import Path  # Added missing import
project_root = Path(__file__).resolve().parent.parent
src_path = str(project_root / "src")
sys.path.append(src_path)


from movie_recommender.data import merge_multiple_datasets
from movie_recommender.utils import( 
    get_data_path,
    save_processed_data
    )

if __name__ == "__main__":
    # Get full paths to raw datasets using get_data_path
    raw_data_dir = get_data_path("raw")
    dataset_paths = [
        raw_data_dir / "row_movie_metadata" / "movies_metadata.csv",
        raw_data_dir / "row_movie_metadata" / "credits.csv",
        raw_data_dir / "row_movie_metadata" / "keywords.csv"
    ]
    
    columns_to_drop = [
        "homepage",
        "tagline",
        "poster_path",
        "adult",
        "status",
        "video",
        "belongs_to_collection",
        "overview",
    ]
    
    merged_data = merge_multiple_datasets(dataset_paths, columns_to_drop)
    
    merged_data_path = get_data_path("merged")
    # Save to merged data directory
    output_path = merged_data_path / "movies_metadata_merged.csv"
    print(f"Saving merged data to: {output_path}")
    save_processed_data(merged_data, output_path)
