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
    valid_types = ['raw', 'processed', 'interim', 'external']
    
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