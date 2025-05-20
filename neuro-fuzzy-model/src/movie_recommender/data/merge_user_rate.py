import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple
import os
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_ratings_movies(
    ratings_path: str,
    movies_path: str,
    output_path: str,
    chunk_size: int = 100000,
    columns_to_keep: Optional[list] = None
) -> Tuple[bool, str]:
    """
    Merge ratings and movies CSV files using movie IDs with chunking for memory efficiency.
    Saves the output as a parquet file for better performance.
    
    Args:
        ratings_path (str): Path to the ratings CSV file
        movies_path (str): Path to the movies CSV file
        output_path (str): Path where the merged parquet file will be saved
        chunk_size (int, optional): Number of rows to process at once. Defaults to 100000.
        columns_to_keep (list, optional): List of columns to keep in the final output.
            If None, keeps all columns. Defaults to None.
    
    Returns:
        Tuple[bool, str]: (Success status, Message)
    """
    try:
        # Validate input files exist
        if not all(os.path.exists(path) for path in [ratings_path, movies_path]):
            missing_files = [path for path in [ratings_path, movies_path] if not os.path.exists(path)]
            raise FileNotFoundError(f"Input files not found: {', '.join(missing_files)}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read movies file (typically smaller, can be loaded entirely)
        logger.info(f"Reading movies file from {movies_path}")
        movies_df = pd.read_csv(movies_path)
        
        # Validate required columns exist
        required_columns = {'movieId', 'title'}
        if not required_columns.issubset(movies_df.columns):
            missing_cols = required_columns - set(movies_df.columns)
            raise ValueError(f"Movies file missing required columns: {missing_cols}")

        # Process ratings in chunks and write to parquet
        logger.info(f"Processing ratings file from {ratings_path} in chunks of {chunk_size}")
        first_chunk = True
        temp_parquet_files = []
        
        for i, chunk in enumerate(pd.read_csv(ratings_path, chunksize=chunk_size)):
            # Merge chunk with movies data
            merged_chunk = pd.merge(
                chunk,
                movies_df,
                on='movieId',
                how='left'
            )
            
            # Select columns if specified
            if columns_to_keep:
                merged_chunk = merged_chunk[columns_to_keep]
            
            # Convert to pyarrow table
            table = pa.Table.from_pandas(merged_chunk)
            
            # Save chunk to temporary parquet file
            temp_file = f"{output_path}.temp_{i}.parquet"
            pq.write_table(table, temp_file)
            temp_parquet_files.append(temp_file)
            
            logger.info(f"Processed chunk {i+1} of {len(chunk)} rows")

        # Combine all temporary parquet files
        logger.info("Combining all chunks into final parquet file...")
        tables = [pq.read_table(f) for f in temp_parquet_files]
        combined_table = pa.concat_tables(tables)
        pq.write_table(combined_table, output_path)
        
        # Clean up temporary files
        for temp_file in temp_parquet_files:
            os.remove(temp_file)

        logger.info(f"Successfully merged files. Output saved to {output_path}")
        return True, "Files merged successfully"

    except Exception as e:
        error_msg = f"Error merging files: {str(e)}"
        logger.error(error_msg)
        # Clean up any temporary files in case of error
        for temp_file in temp_parquet_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return False, error_msg

def merge_with_links(
    parquet_path: str,
    links_path: str,
    output_path: str,
    chunk_size: int = 100000,
    columns_to_keep: Optional[list] = None
) -> Tuple[bool, str]:
    """
    Merge the merged ratings-movies parquet file with links data using movie ID.
    
    Args:
        parquet_path (str): Path to the input parquet file (merged ratings-movies)
        links_path (str): Path to the links CSV file
        output_path (str): Path where the final merged parquet file will be saved
        chunk_size (int, optional): Number of rows to process at once. Defaults to 100000.
        columns_to_keep (list, optional): List of columns to keep in the final output.
            If None, keeps all columns. Defaults to None.
    
    Returns:
        Tuple[bool, str]: (Success status, Message)
    """
    try:
        # Validate input files exist
        if not all(os.path.exists(path) for path in [parquet_path, links_path]):
            missing_files = [path for path in [parquet_path, links_path] if not os.path.exists(path)]
            raise FileNotFoundError(f"Input files not found: {', '.join(missing_files)}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read links file (typically smaller, can be loaded entirely)
        logger.info(f"Reading links file from {links_path}")
        links_df = pd.read_csv(links_path)
        
        # Validate required columns exist
        required_columns = {'movieId', 'imdbId', 'tmdbId'}
        if not required_columns.issubset(links_df.columns):
            missing_cols = required_columns - set(links_df.columns)
            raise ValueError(f"Links file missing required columns: {missing_cols}")

        # Process parquet file in chunks
        logger.info(f"Processing parquet file from {parquet_path} in chunks of {chunk_size}")
        temp_parquet_files = []
        
        # Read parquet file in chunks
        parquet_file = pq.ParquetFile(parquet_path)
        num_row_groups = parquet_file.num_row_groups
        
        for i in range(num_row_groups):
            # Read chunk from parquet
            chunk = parquet_file.read_row_group(i).to_pandas()
            
            # Merge chunk with links data
            merged_chunk = pd.merge(
                chunk,
                links_df,
                on='movieId',
                how='left'
            )
            
            # Select columns if specified
            if columns_to_keep:
                merged_chunk = merged_chunk[columns_to_keep]
            
            # Convert to pyarrow table
            table = pa.Table.from_pandas(merged_chunk)
            
            # Save chunk to temporary parquet file
            temp_file = f"{output_path}.temp_{i}.parquet"
            pq.write_table(table, temp_file)
            temp_parquet_files.append(temp_file)
            
            logger.info(f"Processed chunk {i+1} of {num_row_groups}")

        # Combine all temporary parquet files
        logger.info("Combining all chunks into final parquet file...")
        tables = [pq.read_table(f) for f in temp_parquet_files]
        combined_table = pa.concat_tables(tables)
        pq.write_table(combined_table, output_path)
        
        # Clean up temporary files
        for temp_file in temp_parquet_files:
            os.remove(temp_file)

        logger.info(f"Successfully merged files. Output saved to {output_path}")
        return True, "Files merged successfully"

    except Exception as e:
        error_msg = f"Error merging files: {str(e)}"
        logger.error(error_msg)
        # Clean up any temporary files in case of error
        for temp_file in temp_parquet_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return False, error_msg



