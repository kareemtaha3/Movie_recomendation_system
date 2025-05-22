"""Utility functions for parallel processing to speed up data processing tasks."""

import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ..utils.logging import get_logger

logger = get_logger(__name__)

def get_optimal_workers():
    """
    Get the optimal number of worker processes based on CPU count.
    
    Returns:
        int: Optimal number of worker processes (CPU count - 1, minimum 1)
    """
    # Use one less than available CPUs to avoid system slowdown
    return max(1, cpu_count() - 1)

def parallel_process(items, process_func, n_workers=None, desc="Processing", use_tqdm=True, **kwargs):
    """
    Process items in parallel using multiprocessing.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        n_workers: Number of worker processes (default: optimal based on CPU count)
        desc: Description for progress bar
        use_tqdm: Whether to show progress bar
        **kwargs: Additional arguments to pass to process_func
    
    Returns:
        list: Results from processing all items
    """
    if n_workers is None:
        n_workers = get_optimal_workers()
    
    logger.info(f"Starting parallel processing with {n_workers} workers")
    
    # Create a partial function with the additional kwargs
    if kwargs:
        func = partial(process_func, **kwargs)
    else:
        func = process_func
    
    # Process in parallel
    with Pool(processes=n_workers) as pool:
        if use_tqdm:
            results = list(tqdm(pool.imap(func, items), total=len(items), desc=desc))
        else:
            results = pool.map(func, items)
    
    logger.info(f"Completed parallel processing of {len(items)} items")
    return results

def parallel_dataframe_apply(df, func, column=None, n_workers=None, desc="Processing", **kwargs):
    """
    Apply a function to each row or a specific column of a DataFrame in parallel.
    
    Args:
        df: Pandas DataFrame
        func: Function to apply
        column: Column to apply function to (if None, applies to each row)
        n_workers: Number of worker processes
        desc: Description for progress bar
        **kwargs: Additional arguments to pass to func
    
    Returns:
        list: Results from applying function to each row/value
    """
    if column is not None:
        # Apply to a specific column
        items = df[column].tolist()
    else:
        # Apply to each row as a Series
        items = [row for _, row in df.iterrows()]
    
    return parallel_process(items, func, n_workers, desc, **kwargs)

def chunked_parallel_process(items, process_func, chunk_size=1000, n_workers=None, desc="Processing", **kwargs):
    """
    Process items in parallel using chunking for memory efficiency.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each chunk of items
        chunk_size: Size of each chunk
        n_workers: Number of worker processes
        desc: Description for progress bar
        **kwargs: Additional arguments to pass to process_func
    
    Returns:
        list: Combined results from all chunks
    """
    # Create chunks
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    logger.info(f"Split {len(items)} items into {len(chunks)} chunks of size {chunk_size}")
    
    # Process each chunk in parallel
    chunk_results = parallel_process(chunks, process_func, n_workers, desc, **kwargs)
    
    # Combine results (assuming each result is a list or can be combined)
    return [item for sublist in chunk_results for item in sublist]