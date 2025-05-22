# Configuration utilities

import os
import yaml
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_DIR / 'configs'
DATA_DIR = PROJECT_DIR / 'data'
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'

def load_config(config_name='model_params.yaml'):
    """Load configuration from YAML file.
    
    Args:
        config_name (str): Path to the configuration file. Can be absolute or relative to the configs directory.
        
    Returns:
        dict: Configuration parameters.
    """
    # Try the path as given first
    config_path = Path(config_name)
    if not config_path.is_absolute():
        # If not absolute and doesn't exist, try in the configs directory
        config_path = CONFIG_DIR / config_path.name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_data_path(config=None, data_type='processed'):
    """Get the path to a data directory.
    
    Args:
        config (dict, optional): Configuration parameters. If provided, will check for custom data paths.
        data_type (str): Type of data directory ('raw', 'processed', 'interim', 'external').
        
    Returns:
        Path: Path to the specified data directory.
    """
    valid_types = ['raw', 'processed', 'interim', 'external','merged']
    
    if data_type not in valid_types:
        raise ValueError(f"Invalid data type: {data_type}. Must be one of {valid_types}")
    
    # Check if custom data path is specified in config
    if config and 'data_dir' in config:
        data_path = Path(config['data_dir']) / data_type
    else:
        data_path = DATA_DIR / data_type
    
    if not data_path.exists():
        os.makedirs(data_path, exist_ok=True)
    
    return data_path

def get_artifact_path(config=None, artifact_type='models'):
    """Get the path to an artifacts directory.
    
    Args:
        config (dict, optional): Configuration parameters. If provided, will check for custom artifact paths.
        artifact_type (str): Type of artifact directory ('models', 'figures', 'metrics').
        
    Returns:
        Path: Path to the specified artifacts directory.
    """
    valid_types = ['models', 'figures', 'metrics']
    
    if artifact_type not in valid_types:
        raise ValueError(f"Invalid artifact type: {artifact_type}. Must be one of {valid_types}")
    
    # Check if custom artifact path is specified in config
    if config and 'output_dir' in config:
        artifact_path = Path(config['output_dir']) / artifact_type
    else:
        artifact_path = ARTIFACTS_DIR / artifact_type
    
    if not artifact_path.exists():
        os.makedirs(artifact_path, exist_ok=True)
    
    return artifact_path



def save_processed_data(df, output_path):
    """Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the DataFrame
        
    Returns:
        bool: True if successful, raises exception otherwise
    """
    try:
        # Convert to absolute path and normalize path separators for Windows
        output_path = os.path.abspath(os.path.normpath(output_path))
        output_dir = os.path.dirname(output_path)
        
        print(f"Saving to directory: {output_dir} (exists: {os.path.exists(output_dir)})")
        print(f"Full output path: {output_path}")
        
        # Ensure directory exists with multiple attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                os.makedirs(output_dir, exist_ok=True)
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                print(f"Attempt {attempt+1} failed: {e}. Retrying...")
        
        # Save the data
        df.to_csv(output_path, index=False)
        print(f"Successfully saved data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        raise