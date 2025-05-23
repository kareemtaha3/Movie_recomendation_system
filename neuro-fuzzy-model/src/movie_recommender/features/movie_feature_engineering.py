import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple, List, Optional, Union
import logging
import gc
import os
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability - only import if environment variable is set
USE_GPU = os.environ.get('USE_GPU', 'False').lower() in ('true', '1', 't')
HAS_GPU = False

if USE_GPU:
    try:
        import torch
        HAS_GPU = torch.cuda.is_available()
        if HAS_GPU:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not installed. GPU acceleration not available.")

def engineer_movie_features(
    df: pd.DataFrame,
    top_n_companies: int = 10,
    top_n_actors: int = 5,
    batch_size: int = 5000,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Engineer movie/item features for deep learning models with optional GPU acceleration.

    Parameters:
    - df: Input DataFrame with columns as provided.
    - top_n_companies: Number of top production companies to encode.
    - top_n_actors: Number of top actors to consider for popularity.
    - batch_size: Number of rows to process at once (prevents memory errors).
    - use_gpu: Whether to use GPU acceleration if available.

    Returns:
    - DataFrame with new feature columns appended.
    """
    start_time = time.time()
    
    # Check if GPU should and can be used
    use_gpu_flag = use_gpu and HAS_GPU
    if use_gpu and not HAS_GPU:
        logger.warning("GPU requested but not available. Falling back to CPU processing.")
    
    # Process in batches to avoid memory errors
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Processing {total_rows} rows in {num_batches} batches of size {batch_size}")
    
    # Initialize an empty list to store processed batches
    processed_batches = []
    
    # Get all genres first (needed for consistent one-hot encoding across batches)
    logger.info("Preparing genre encoding...")
    all_genres_list = df['genres'].str.split('|').tolist()
    mlb_genres = MultiLabelBinarizer()
    mlb_genres.fit(all_genres_list)  # Just fit, don't transform yet
    
    # Get top companies across the entire dataset
    logger.info("Finding top production companies...")
    all_companies_list = df['production_companies'].str.split('|').tolist()
    all_companies = [comp for sub in all_companies_list if isinstance(sub, list) for comp in sub]
    top_companies = pd.Series(all_companies).value_counts().head(top_n_companies).index
    
    # Process each batch
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rows)
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} (rows {start_idx}-{end_idx})")
        
        # Get the current batch
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Process the batch
        if use_gpu_flag:
            try:
                import torch
                processed_batch = _process_batch_gpu(batch_df, mlb_genres, top_companies, top_n_companies, top_n_actors)
            except Exception as e:
                logger.error(f"Error in GPU processing: {str(e)}. Falling back to CPU.")
                processed_batch = _process_batch_cpu(batch_df, mlb_genres, top_companies, top_n_companies, top_n_actors)
        else:
            processed_batch = _process_batch_cpu(batch_df, mlb_genres, top_companies, top_n_companies, top_n_actors)
        
        processed_batches.append(processed_batch)
        
        # Force garbage collection to free memory
        gc.collect()
        
    # Combine all processed batches
    logger.info("Combining all processed batches")
    engineered = pd.concat(processed_batches, axis=0)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed_time:.2f} seconds")
    
    return engineered

def _process_batch_cpu(
    batch_df: pd.DataFrame,
    mlb_genres: MultiLabelBinarizer,
    top_companies: pd.Index,
    top_n_companies: int,
    top_n_actors: int
) -> pd.DataFrame:
    """
    Process a batch of data using CPU with optimized memory usage.
    """
    # 1. Release year - handle missing dates
    batch_df['release_year'] = pd.to_datetime(batch_df['release_date'], errors='coerce').dt.year
    batch_df['release_year'] = batch_df['release_year'].fillna(-1).astype(int)

    # 2. Runtime normalization - handle zero/missing values
    runtime_mean = batch_df['runtime'].replace(0, np.nan).mean()
    runtime_std = batch_df['runtime'].replace(0, np.nan).std()
    # Avoid division by zero
    if runtime_std == 0 or pd.isna(runtime_std):
        runtime_std = 1.0
    batch_df['runtime_norm'] = (batch_df['runtime'] - runtime_mean) / runtime_std
    batch_df['runtime_norm'] = batch_df['runtime_norm'].fillna(0)

    # 3. Log-scale budget & revenue - handle zeros
    batch_df['log_budget'] = np.log1p(batch_df['budget'].fillna(0))
    batch_df['log_revenue'] = np.log1p(batch_df['revenue'].fillna(0))

    # 4. Normalized popularity & vote_average - handle edge cases
    pop_min = batch_df['popularity'].min()
    pop_max = batch_df['popularity'].max()
    pop_range = pop_max - pop_min
    if pop_range == 0 or pd.isna(pop_range):
        pop_range = 1.0
    batch_df['popularity_norm'] = (batch_df['popularity'] - pop_min) / pop_range
    
    vote_min = batch_df['vote_average'].min()
    vote_max = batch_df['vote_average'].max()
    vote_range = vote_max - vote_min
    if vote_range == 0 or pd.isna(vote_range):
        vote_range = 1.0
    batch_df['vote_average_norm'] = (batch_df['vote_average'] - vote_min) / vote_range

    # 5. Log vote count
    batch_df['log_vote_count'] = np.log1p(batch_df['vote_count'].fillna(0))

    # 6. Language code - handle missing values
    batch_df['language_code'] = batch_df['original_language'].fillna('').astype('category').cat.codes

    # 7. Genres multi-hot - optimized to reduce memory usage
    batch_df['genres_list'] = batch_df['genres'].fillna('').str.split('|')
    genres_mh = mlb_genres.transform(batch_df['genres_list'])
    genres_df = pd.DataFrame(
        genres_mh,
        columns=[f'genre_{g}' for g in mlb_genres.classes_],
        index=batch_df.index
    )

    # 8. Production companies encoding (top N) - optimized
    batch_df['companies_list'] = batch_df['production_companies'].fillna('').str.split('|')
    
    # More efficient company code assignment
    company_codes = []
    for comps in batch_df['companies_list']:
        if not isinstance(comps, list) or not comps or comps[0] == '':
            company_codes.append(top_n_companies)
        else:
            found = False
            for i, c in enumerate(top_companies):
                if c in comps:
                    company_codes.append(i)
                    found = True
                    break
            if not found:
                company_codes.append(top_n_companies)
    
    batch_df['company_code'] = company_codes

    # 9. Cast list and popularity - optimized to reduce memory usage
    batch_df['cast_list'] = batch_df['cast'].fillna('').str.split('|')
    
    # More efficient actor popularity calculation
    actor_pop_dict = {}
    for _, row in batch_df[['cast_list', 'popularity']].iterrows():
        if isinstance(row['cast_list'], list) and row['cast_list'] and row['cast_list'][0] != '':
            for actor in row['cast_list']:
                if actor in actor_pop_dict:
                    actor_pop_dict[actor][0] += row['popularity']
                    actor_pop_dict[actor][1] += 1
                else:
                    actor_pop_dict[actor] = [row['popularity'], 1]
    
    # Calculate mean popularity
    actor_pop = {actor: values[0]/values[1] for actor, values in actor_pop_dict.items()}
    
    # Vectorized calculation for cast popularity
    cast_popularity = []
    for lst in batch_df['cast_list']:
        if isinstance(lst, list) and lst and lst[0] != '':
            pop_values = [actor_pop.get(actor, 0) for actor in lst]
            cast_popularity.append(np.mean(pop_values) if pop_values else 0)
        else:
            cast_popularity.append(0)
    
    batch_df['cast_popularity'] = cast_popularity

    # 10. Director features - handle missing values
    batch_df['director_name'] = batch_df['crew'].fillna('')  # Already processed to just director name
    batch_df['director_code'] = batch_df['director_name'].astype('category').cat.codes
    
    # Director popularity: total vote_count across their movies
    dir_pop = batch_df.groupby('director_name')['vote_count'].sum()
    batch_df['director_popularity'] = batch_df['director_name'].map(dir_pop).fillna(0)

    # Combine all engineered features
    engineered = pd.concat([batch_df, genres_df], axis=1)

    return engineered

def _process_batch_gpu(
    batch_df: pd.DataFrame,
    mlb_genres: MultiLabelBinarizer,
    top_companies: pd.Index,
    top_n_companies: int,
    top_n_actors: int
) -> pd.DataFrame:
    """
    Process a batch of data using GPU acceleration with PyTorch.
    """
    import torch
    
    # Most operations are still more reliable on CPU, so we'll only use GPU for the most intensive parts
    
    # 1. Release year - handle missing dates (CPU operation)
    batch_df['release_year'] = pd.to_datetime(batch_df['release_date'], errors='coerce').dt.year
    batch_df['release_year'] = batch_df['release_year'].fillna(-1).astype(int)

    # 2-5. Numeric operations - these can benefit from GPU
    # Convert numeric columns to PyTorch tensors
    numeric_cols = ['runtime', 'budget', 'revenue', 'popularity', 'vote_average', 'vote_count']
    numeric_data = {col: torch.tensor(batch_df[col].fillna(0).values, device='cuda', dtype=torch.float32) 
                   for col in numeric_cols}
    
    # Runtime normalization
    runtime_mean = float(numeric_data['runtime'].mean().item())
    runtime_std = float(numeric_data['runtime'].std().item())
    if runtime_std == 0:
        runtime_std = 1.0
    batch_df['runtime_norm'] = ((numeric_data['runtime'] - runtime_mean) / runtime_std).cpu().numpy()

    # Log-scale budget & revenue
    batch_df['log_budget'] = torch.log1p(numeric_data['budget']).cpu().numpy()
    batch_df['log_revenue'] = torch.log1p(numeric_data['revenue']).cpu().numpy()

    # Normalized popularity & vote_average
    pop_min = float(numeric_data['popularity'].min().item())
    pop_max = float(numeric_data['popularity'].max().item())
    pop_range = pop_max - pop_min
    if pop_range == 0:
        pop_range = 1.0
    batch_df['popularity_norm'] = ((numeric_data['popularity'] - pop_min) / pop_range).cpu().numpy()
    
    vote_min = float(numeric_data['vote_average'].min().item())
    vote_max = float(numeric_data['vote_average'].max().item())
    vote_range = vote_max - vote_min
    if vote_range == 0:
        vote_range = 1.0
    batch_df['vote_average_norm'] = ((numeric_data['vote_average'] - vote_min) / vote_range).cpu().numpy()

    # Log vote count
    batch_df['log_vote_count'] = torch.log1p(numeric_data['vote_count']).cpu().numpy()

    # 6. Language code - categorical operations (CPU operation)
    batch_df['language_code'] = batch_df['original_language'].fillna('').astype('category').cat.codes

    # 7. Genres multi-hot - GPU acceleration for transformation
    batch_df['genres_list'] = batch_df['genres'].fillna('').str.split('|')
    genres_mh = mlb_genres.transform(batch_df['genres_list'])
    
    # Move to GPU for faster processing
    genres_tensor = torch.tensor(genres_mh, device='cuda')
    # Move back to CPU for DataFrame creation
    genres_mh = genres_tensor.cpu().numpy()
    genres_df = pd.DataFrame(
        genres_mh,
        columns=[f'genre_{g}' for g in mlb_genres.classes_],
        index=batch_df.index
    )

    # 8-10. Remaining operations - do on CPU as they're more complex
    # Production companies encoding
    batch_df['companies_list'] = batch_df['production_companies'].fillna('').str.split('|')
    
    # More efficient company code assignment
    company_codes = []
    for comps in batch_df['companies_list']:
        if not isinstance(comps, list) or not comps or comps[0] == '':
            company_codes.append(top_n_companies)
        else:
            found = False
            for i, c in enumerate(top_companies):
                if c in comps:
                    company_codes.append(i)
                    found = True
                    break
            if not found:
                company_codes.append(top_n_companies)
    
    batch_df['company_code'] = company_codes

    # Cast list and popularity
    batch_df['cast_list'] = batch_df['cast'].fillna('').str.split('|')
    
    # Actor popularity calculation
    actor_pop_dict = {}
    for _, row in batch_df[['cast_list', 'popularity']].iterrows():
        if isinstance(row['cast_list'], list) and row['cast_list'] and row['cast_list'][0] != '':
            for actor in row['cast_list']:
                if actor in actor_pop_dict:
                    actor_pop_dict[actor][0] += row['popularity']
                    actor_pop_dict[actor][1] += 1
                else:
                    actor_pop_dict[actor] = [row['popularity'], 1]
    
    # Calculate mean popularity
    actor_pop = {actor: values[0]/values[1] for actor, values in actor_pop_dict.items()}
    
    # Vectorized calculation for cast popularity
    cast_popularity = []
    for lst in batch_df['cast_list']:
        if isinstance(lst, list) and lst and lst[0] != '':
            pop_values = [actor_pop.get(actor, 0) for actor in lst]
            cast_popularity.append(np.mean(pop_values) if pop_values else 0)
        else:
            cast_popularity.append(0)
    
    batch_df['cast_popularity'] = cast_popularity

    # Director features
    batch_df['director_name'] = batch_df['crew'].fillna('')
    batch_df['director_code'] = batch_df['director_name'].astype('category').cat.codes
    
    # Director popularity
    dir_pop = batch_df.groupby('director_name')['vote_count'].sum()
    batch_df['director_popularity'] = batch_df['director_name'].map(dir_pop).fillna(0)

    # Combine all engineered features
    engineered = pd.concat([batch_df, genres_df], axis=1)
    
    # Clean up GPU memory
    del numeric_data, genres_tensor
    torch.cuda.empty_cache()
    
    return engineered

# Example usage:
# To use GPU acceleration:
# import os
# os.environ['USE_GPU'] = 'True'  # Set this before importing the module
# engineered_df = engineer_movie_features(your_dataframe, batch_size=5000, use_gpu=True)
#
# To use CPU only:
# engineered_df = engineer_movie_features(your_dataframe, batch_size=5000, use_gpu=False)
