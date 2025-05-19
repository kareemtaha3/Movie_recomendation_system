import logging
import os
import sys
from pathlib import Path


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration for the project.
    
    Parameters
    ----------
    log_file : str, optional
        Path to the log file. If None, logs will only be printed to console.
    log_level : int, optional
        Logging level. Default is logging.INFO.
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger('movie_recommender')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers to avoid duplicates
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance. If the logger doesn't exist, it will be created.
    
    Parameters
    ----------
    name : str, optional
        Name of the logger. If None, the root logger will be returned.
        
    Returns
    -------
    logger : logging.Logger
        Logger instance.
    """
    if name:
        return logging.getLogger(f'movie_recommender.{name}')
    else:
        return logging.getLogger('movie_recommender')


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    
    Parameters
    ----------
    func : callable
        Function to be decorated.
        
    Returns
    -------
    wrapper : callable
        Decorated function.
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper