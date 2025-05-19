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
        config_name (str): Name of the configuration file in the configs directory.
        
    Returns:
        dict: Configuration parameters.
    """
    config_path = CONFIG_DIR / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_data_path(data_type='processed'):
    """Get the path to a data directory.
    
    Args:
        data_type (str): Type of data directory ('raw', 'processed', 'interim', 'external').
        
    Returns:
        Path: Path to the specified data directory.
    """
    valid_types = ['raw', 'processed', 'interim', 'external','merged']
    
    if data_type not in valid_types:
        raise ValueError(f"Invalid data type: {data_type}. Must be one of {valid_types}")
    
    data_path = DATA_DIR / data_type
    
    if not data_path.exists():
        os.makedirs(data_path, exist_ok=True)
    
    return data_path

def get_artifact_path(artifact_type='models'):
    """Get the path to an artifacts directory.
    
    Args:
        artifact_type (str): Type of artifact directory ('models', 'figures', 'metrics').
        
    Returns:
        Path: Path to the specified artifacts directory.
    """
    valid_types = ['models', 'figures', 'metrics']
    
    if artifact_type not in valid_types:
        raise ValueError(f"Invalid artifact type: {artifact_type}. Must be one of {valid_types}")
    
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
                # Create all parent directories if they don't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Verify directory was created successfully
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    print(f"Directory created/verified: {output_dir} (exists: True)")
                    # Check if directory is writable
                    if os.access(output_dir, os.W_OK):
                        print(f"Directory is writable: {output_dir}")
                    else:
                        print(f"WARNING: Directory is not writable: {output_dir}")
                    break
                else:
                    raise FileNotFoundError(f"Failed to create directory: {output_dir}")
            except Exception as dir_err:
                if attempt < max_attempts - 1:
                    print(f"Attempt {attempt+1}/{max_attempts} to create directory failed: {str(dir_err)}")
                    import time
                    time.sleep(1)  # Wait before retrying
                else:
                    raise
        
        # Save data with retry mechanism
        print(f"DataFrame shape before saving: {df.shape}")
        max_save_attempts = 3
        for save_attempt in range(max_save_attempts):
            try:
                # Flush any pending file operations before saving
                import gc
                gc.collect()
                
                # Create a temporary filename in the same directory
                temp_path = f"{output_path}.temp"
                
                # Save to temporary file first
                print(f"Saving to temporary file: {temp_path}")
                df.to_csv(temp_path, index=False)
                
                # Force sync to disk
                import gc
                gc.collect()
                
                # Verify temp file was created
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    # Rename temp file to final filename (atomic operation on most file systems)
                    print(f"Renaming {temp_path} to {output_path}")
                    if os.path.exists(output_path):
                        os.remove(output_path)  # Remove existing file if it exists
                    os.rename(temp_path, output_path)
                else:
                    raise FileNotFoundError(f"Temporary file was not created or is empty: {temp_path}")
                
                # Wait a moment to ensure file is fully written
                import time
                time.sleep(0.5)
                
                # Verify final file was created
                file_exists = os.path.exists(output_path)
                file_size = os.path.getsize(output_path) if file_exists else 0
                print(f"Processed data saved to: {output_path} (exists: {file_exists}, size: {file_size} bytes)")
                
                if not file_exists or file_size == 0:
                    raise FileNotFoundError(f"File was not created or is empty: {output_path}")
                
                # Ensure file is properly closed and synced to disk
                gc.collect()
                
                return True
            except Exception as save_err:
                if save_attempt < max_save_attempts - 1:
                    print(f"Attempt {save_attempt+1}/{max_save_attempts} to save file failed: {str(save_err)}")
                    import time
                    time.sleep(1)  # Wait before retrying
                else:
                    raise
    except Exception as e:
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Error saving processed data: {e}")