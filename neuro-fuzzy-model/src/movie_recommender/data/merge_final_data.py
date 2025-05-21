import os
import pandas as pd
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import ast  # Add this for safely evaluating string literals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_cast(cast_str):
    """
    Process cast string to extract first 5 cast members' names.
    
    Parameters:
    -----------
    cast_str : str
        String representation of cast list
    
    Returns:
    --------
    str
        Pipe-separated string of first 5 cast members' names
    """
    try:
        if pd.isna(cast_str):
            return ""
        
        # Convert string representation to actual list
        cast_list = ast.literal_eval(cast_str)
        
        # Extract first 5 cast members' names
        cast_names = [cast_member['name'] for cast_member in cast_list[:5]]
        
        # Join with pipe separator
        return "|".join(cast_names)
    except Exception as e:
        logger.warning(f"Error processing cast data: {e}")
        return ""

def process_crew(crew_str):
    """
    Process crew string to extract director's name.
    
    Parameters:
    -----------
    crew_str : str
        String representation of crew list
    
    Returns:
    --------
    str
        Director's name or empty string if not found
    """
    try:
        if pd.isna(crew_str):
            return ""
        
        # Convert string representation to actual list
        crew_list = ast.literal_eval(crew_str)
        
        # Find the director
        for crew_member in crew_list:
            if crew_member.get('job') == 'Director':
                return crew_member['name']
        
        return ""  # Return empty string if no director found
    except Exception as e:
        logger.warning(f"Error processing crew data: {e}")
        return ""

def process_production_companies(companies_str):
    """
    Process production companies string to extract company names.
    
    Parameters:
    -----------
    companies_str : str
        String representation of production companies list
    
    Returns:
    --------
    str
        Pipe-separated string of company names
    """
    try:
        if pd.isna(companies_str):
            return ""
        
        # Convert string representation to actual list
        companies_list = ast.literal_eval(companies_str)
        
        # Extract company names
        company_names = [company['name'] for company in companies_list]
        
        # Join with pipe separator
        return "|".join(company_names)
    except Exception as e:
        logger.warning(f"Error processing production companies data: {e}")
        return ""

def merge_final_data(data_dir="data", chunk_size=500_000):
    """
    Merge final_user_data.parquet and movies_metadata_merged.csv using pandas merge
    with chunked processing and temporary files for better memory management.
    
    Parameters:
    -----------
    data_dir : str
        Base directory containing the data files
    chunk_size : int
        Number of rows to process in each chunk
    """
    try:
        # Configure paths
        base_path = Path(data_dir)
        merged_dir = base_path / "merged"
        interim_dir = base_path / "interim"
        
        # Ensure output directory exists
        interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        movie_path = merged_dir / "movies_metadata_merged.csv"
        user_path = merged_dir / "final_user_data.parquet"
        output_path = interim_dir / "final_merged_data.parquet"
        
        # Validate input files
        if not movie_path.exists():
            raise FileNotFoundError(f"Movie metadata file not found: {movie_path}")
        if not user_path.exists():
            raise FileNotFoundError(f"User data file not found: {user_path}")
        
        # Load and prepare movie metadata
        logger.info(f"Loading movie metadata from {movie_path}")
        movie_df = pd.read_csv(movie_path)
        
        # Select only required columns
        columns_to_keep = [
            'imdb_id',
            'budget',
            'original_language',
            'popularity',
            'release_date',
            'revenue',
            'runtime',
            'vote_average',
            'vote_count',
            'cast',
            'crew',
            'production_companies'  # Add production companies column
        ]
        
        # Verify all required columns exist
        missing_columns = [col for col in columns_to_keep if col not in movie_df.columns]
        if missing_columns:
            raise ValueError(f"Movie metadata missing required columns: {missing_columns}")
        
        # Keep only required columns
        movie_df = movie_df[columns_to_keep]
        
        # Process cast, crew, and production companies data
        logger.info("Processing cast, crew, and production companies data...")
        movie_df['cast'] = movie_df['cast'].apply(process_cast)
        movie_df['crew'] = movie_df['crew'].apply(process_crew)
        movie_df['production_companies'] = movie_df['production_companies'].apply(process_production_companies)
        logger.info("Data processing completed")
        
        # Clean and prepare movie metadata
        movie_df['imdb_id'] = movie_df['imdb_id'].astype(str).str.replace('.0', '', regex=False)
        movie_df = movie_df.drop_duplicates(subset=['imdb_id'], keep='first')
        
        # Log movie metadata info
        logger.info(f"Loaded movie metadata: {len(movie_df):,} unique movies")
        logger.info(f"Columns in movie metadata: {movie_df.columns.tolist()}")
        logger.info(f"Sample of movie metadata imdb_ids: {movie_df['imdb_id'].head().tolist()}")
        
        # Open parquet file
        parquet_file = pq.ParquetFile(user_path)
        num_row_groups = parquet_file.num_row_groups
        total_rows = parquet_file.metadata.num_rows
        
        logger.info(f"Processing {total_rows:,} rows from {num_row_groups} row groups")
        
        # Process in chunks with progress bar
        total_processed = 0
        temp_parquet_files = []
        
        with tqdm(total=total_rows, desc="Processing Rows", unit="rows") as pbar:
            for rg in range(num_row_groups):
                # Read row group
                df_chunk = parquet_file.read_row_group(rg).to_pandas()
                
                # Ensure imdbId is string type
                df_chunk['imdbId'] = df_chunk['imdbId'].astype(str)
                
                # Log chunk info
                logger.info(f"Processing row group {rg + 1}/{num_row_groups}")
                logger.info(f"Chunk shape before merge: {df_chunk.shape}")
                
                # Merge with movie metadata
                merged_chunk = pd.merge(
                    df_chunk,
                    movie_df,
                    left_on='imdbId',
                    right_on='imdb_id',
                    how='inner'
                )
                
                # Log merge results
                logger.info(f"Chunk shape after merge: {merged_chunk.shape}")
                
                # Convert to pyarrow table
                table = pa.Table.from_pandas(merged_chunk)
                
                # Save chunk to temporary parquet file
                temp_file = f"{output_path}.temp_{rg}.parquet"
                pq.write_table(
                    table,
                    temp_file,
                    compression='zstd',
                    write_statistics=True
                )
                temp_parquet_files.append(temp_file)
                
                # Update progress
                chunk_size = len(merged_chunk)
                total_processed += chunk_size
                pbar.update(chunk_size)
                
                # Log progress
                logger.info(f"Processed {total_processed:,} rows so far")
        
        # Combine all temporary parquet files
        logger.info("Combining all chunks into final parquet file...")
        tables = [pq.read_table(f) for f in temp_parquet_files]
        combined_table = pa.concat_tables(tables)
        pq.write_table(
            combined_table,
            output_path,
            compression='zstd',
            write_statistics=True
        )
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for temp_file in temp_parquet_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logger.info(f"Completed processing {total_processed:,} rows")
        logger.info(f"Output saved to: {output_path}")
        
        # Return a tuple with success status and count
        return True, total_processed
        
    except Exception as e:
        error_msg = f"Error merging files: {str(e)}"
        logger.error(error_msg)
        # Clean up any temporary files in case of error
        for temp_file in temp_parquet_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return False, 0
