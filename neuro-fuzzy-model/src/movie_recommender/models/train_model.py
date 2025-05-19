# Model training script

import os
import logging
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import skfuzzy as fuzz
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
MODELS_DIR = PROJECT_DIR / 'artifacts' / 'models'
METRICS_DIR = PROJECT_DIR / 'artifacts' / 'metrics'
CONFIG_PATH = PROJECT_DIR / 'configs' / 'model_params.yaml'

def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_processed_data():
    """Load processed data for model training."""
    logger.info(f"Loading processed data from {PROCESSED_DATA_DIR}")
    
    try:
        # Load user-item matrix
        user_item_matrix_path = PROCESSED_DATA_DIR / 'user_item_matrix.csv'
        movies_path = PROCESSED_DATA_DIR / 'processed_movies.csv'
        
        if not user_item_matrix_path.exists() or not movies_path.exists():
            logger.warning("Processed data files not found. Run data preparation first.")
            return None, None
        
        # Load data
        user_item_df = pd.read_csv(user_item_matrix_path, index_col=0)
        movies_df = pd.read_csv(movies_path)
        
        logger.info(f"Loaded user-item matrix with shape {user_item_df.shape} and {len(movies_df)} movies")
        return user_item_df, movies_df
        
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None, None

def prepare_training_data(user_item_df, config):
    """Prepare data for model training."""
    if user_item_df is None:
        logger.error("Cannot prepare training data: missing input data")
        return None, None, None, None
    
    logger.info("Preparing training data...")
    
    try:
        # Convert to numpy array
        user_item_matrix = user_item_df.values
        
        # Create user and item indices
        user_indices = np.arange(user_item_matrix.shape[0])
        item_indices = np.arange(user_item_matrix.shape[1])
        
        # Create training data
        # For each non-zero entry in the matrix, create a training example
        user_idx, item_idx = user_item_matrix.nonzero()
        ratings = user_item_matrix[user_idx, item_idx]
        
        # Normalize ratings to [0, 1] range
        ratings_normalized = ratings / 5.0  # Assuming ratings are on a 1-5 scale
        
        # Create feature vectors
        X = np.column_stack((user_idx, item_idx))
        y = ratings_normalized
        
        # Split into training and validation sets
        test_size = config['data']['test_size']
        random_state = config['data']['random_state']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Prepared {len(X_train)} training samples and {len(X_val)} validation samples")
        
        # Save user and item indices mapping for later use
        user_mapping = {i: user_id for i, user_id in enumerate(user_item_df.index)}
        item_mapping = {i: item_id for i, item_id in enumerate(user_item_df.columns)}
        
        mappings = {
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'n_users': len(user_indices),
            'n_items': len(item_indices)
        }
        
        return X_train, X_val, y_train, y_val, mappings
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return None, None, None, None, None

def create_neuro_fuzzy_model(n_users, n_items, config):
    """Create a neuro-fuzzy model for recommendation."""
    logger.info("Creating neuro-fuzzy model...")
    
    try:
        # Extract model parameters from config
        nn_config = config['model']['neural_network']
        architecture = nn_config['architecture']
        activation = nn_config['activation']
        dropout_rate = nn_config['dropout_rate']
        learning_rate = nn_config['learning_rate']
        
        # Create user embedding input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = tf.keras.layers.Embedding(
            input_dim=n_users,
            output_dim=architecture[0],
            name='user_embedding'
        )(user_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)
        
        # Create item embedding input
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=architecture[0],
            name='item_embedding'
        )(item_input)
        item_embedding = tf.keras.layers.Flatten()(item_embedding)
        
        # Combine embeddings
        concat = Concatenate()([user_embedding, item_embedding])
        
        # Add dense layers
        dense = concat
        for units in architecture:
            dense = Dense(units, activation=activation)(dense)
            dense = Dropout(dropout_rate)(dense)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='prediction')(dense)
        
        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info(f"Created neuro-fuzzy model with architecture: {architecture}")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return None

def train_model(model, X_train, y_train, X_val, y_val, config):
    """Train the neuro-fuzzy model."""
    if model is None or X_train is None or y_train is None:
        logger.error("Cannot train model: missing model or training data")
        return None, None
    
    logger.info("Training model...")
    
    try:
        # Extract training parameters from config
        nn_config = config['model']['neural_network']
        batch_size = nn_config['batch_size']
        epochs = nn_config['epochs']
        patience = nn_config['early_stopping_patience']
        
        # Prepare input data
        user_train = X_train[:, 0].astype(int)
        item_train = X_train[:, 1].astype(int)
        
        user_val = X_val[:, 0].astype(int)
        item_val = X_val[:, 1].astype(int)
        
        # Create callbacks
        os.makedirs(MODELS_DIR, exist_ok=True)
        checkpoint_path = MODELS_DIR / 'best_model.h5'
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            [user_train, item_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([user_val, item_val], y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        final_train_loss = history.history['loss'][-1]
        final_train_mae = history.history['mae'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        
        metrics = {
            'train_loss': float(final_train_loss),
            'train_mae': float(final_train_mae),
            'val_loss': float(final_val_loss),
            'val_mae': float(final_val_mae),
            'epochs_trained': len(history.history['loss'])
        }
        
        logger.info(f"Model training completed. Final validation MAE: {final_val_mae:.4f}")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None

def save_model_and_metrics(model, metrics, mappings):
    """Save the trained model, metrics, and mappings."""
    if model is None or metrics is None or mappings is None:
        logger.error("Cannot save: missing model, metrics, or mappings")
        return False
    
    logger.info(f"Saving model and metrics...")
    
    try:
        # Create directories if they don't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        # Save model
        model_path = MODELS_DIR / 'final_model.h5'
        model.save(str(model_path))
        
        # Save mappings
        mappings_path = MODELS_DIR / 'mappings.joblib'
        joblib.dump(mappings, mappings_path)
        
        # Save metrics
        metrics_path = METRICS_DIR / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model and metrics saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model and metrics: {e}")
        return False

def main():
    """Main function to run the model training pipeline."""
    logger.info("Starting model training pipeline")
    
    # Load configuration
    config = load_config()
    
    # Load processed data
    user_item_df, movies_df = load_processed_data()
    
    # Prepare training data
    X_train, X_val, y_train, y_val, mappings = prepare_training_data(user_item_df, config)
    
    if X_train is None or mappings is None:
        logger.error("Failed to prepare training data. Exiting.")
        return
    
    # Create model
    n_users = mappings['n_users']
    n_items = mappings['n_items']
    model = create_neuro_fuzzy_model(n_users, n_items, config)
    
    if model is None:
        logger.error("Failed to create model. Exiting.")
        return
    
    # Train model
    trained_model, metrics = train_model(model, X_train, y_train, X_val, y_val, config)
    
    if trained_model is None:
        logger.error("Failed to train model. Exiting.")
        return
    
    # Save model and metrics
    success = save_model_and_metrics(trained_model, metrics, mappings)
    
    if success:
        logger.info("Model training pipeline completed successfully")
    else:
        logger.error("Model training pipeline failed at the saving stage")

if __name__ == "__main__":
    main()