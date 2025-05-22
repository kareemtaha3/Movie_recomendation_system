# Neuro-Fuzzy Movie Recommendation System

This project implements a movie recommendation system using a hybrid neuro-fuzzy approach, combining neural networks with fuzzy logic to provide personalized movie recommendations.

## Project Structure

```
neuro-fuzzy-model/
├── artifacts/           # Stores model artifacts
│   ├── figures/         # Visualization outputs
│   ├── metrics/         # Evaluation metrics
│   └── models/          # Trained models
├── configs/             # Configuration files
├── data/                # Data directory
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
├── logs/                # Log files
├── pipelines/           # Pipeline scripts
├── src/                 # Source code
│   └── movie_recommender/
│       ├── data/        # Data processing modules
│       ├── features/    # Feature engineering modules
│       ├── models/      # Model implementation
│       ├── utils/       # Utility functions
│       └── visualization/ # Visualization modules
├── prepare_data.py      # Data preparation script
└── run_training_pipeline_fixed.py  # Training pipeline runner
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: tensorflow, scikit-fuzzy, pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install tensorflow scikit-fuzzy pandas numpy matplotlib seaborn scikit-learn tqdm joblib pyyaml pyarrow
```

### Data Preparation

Place your raw data files in the `data/raw/` directory. The system expects the following files:

- `movies.csv`: Movie information with columns including movieId, title, and genres
- `ratings.csv`: User ratings with columns including userId, movieId, and rating

Run the data preparation script to process the raw data and generate features:

```bash
python prepare_data.py --log-level INFO
```

Options:

- `--config`: Path to the configuration file (default: configs/model_params.yaml)
- `--log-level`: Logging level (default: INFO)
- `--force`: Force reprocessing of data even if processed data exists
- `--output-dir`: Directory to save processed features

### Training the Model

Run the training pipeline to train the neuro-fuzzy recommendation model:

```bash
python run_training_pipeline_fixed.py --log-level INFO
```

Options:

- `--config`: Path to the configuration file (default: configs/model_params.yaml)
- `--log-level`: Logging level (default: INFO)
- `--features-dir`: Directory containing processed features
- `--output-dir`: Directory to save trained models and artifacts
- `--no-eval`: Skip evaluation step
- `--no-plots`: Skip generating plots

## Model Architecture

The recommendation system uses a hybrid approach combining:

1. **Neural Network Component**: A deep neural network that learns patterns from user-movie interactions and features
2. **Fuzzy Logic Component**: A fuzzy inference system that incorporates domain knowledge and handles uncertainty

The model combines predictions from both components to generate final recommendations.

## Key Features

- Hybrid neuro-fuzzy architecture for improved recommendation accuracy
- Comprehensive feature engineering for movies and users
- Parallel processing for efficient data handling
- Detailed evaluation metrics and visualizations
- Model persistence for easy deployment

## Improvements Made

1. Fixed the fuzzy feature calculation to use actual genre and rating similarity instead of random values
2. Enhanced model training with early stopping to prevent overfitting
3. Improved error handling and logging throughout the pipeline
4. Added proper artifact storage for models, metrics, and visualizations
5. Implemented parallel processing for efficient data handling
6. Added comprehensive documentation and usage instructions

## Evaluation Metrics

The model is evaluated using the following metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Correlation coefficient
- Binary classification metrics (accuracy, precision, recall, F1 score)

Visualization outputs include:

- Rating distribution
- Predicted vs actual ratings
- Error distribution
- Confusion matrix
- Training history
- Fuzzy membership functions

## Configuration

The model and training parameters can be configured in the `configs/model_params.yaml` file. Key parameters include:

- Neural network architecture
- Fuzzy system parameters
- Training parameters (learning rate, batch size, epochs)
- Evaluation parameters

## Optimized Feature Engineering and Training Pipelines

The feature engineering and training pipelines have been optimized for performance and reliability with the following improvements:

- **Parallel Processing**: Utilizes multiple CPU cores to speed up data processing
- **Memory Efficiency**: Processes data in chunks to handle large datasets
- **Error Handling**: Robust error handling to prevent pipeline failures
- **Progress Tracking**: Visual progress indicators during long-running operations

## Project Structure

The project follows a structured organization:

```
neuro-fuzzy-model/
├── .gitignore           # Git ignore file
├── .dvcignore           # DVC ignore file
├── dvc.yaml             # DVC pipeline definition
├── dvc.lock             # DVC pipeline state (auto-generated)
├── .dvc/                # DVC internal files
├── README.md            # Project documentation
├── environment.yml      # Conda environment specification
├── Makefile             # Common commands
├── configs/             # Configuration files
├── data/                # Data directories
│   ├── external/        # Original data from external sources
│   ├── raw/             # Initial cleaned data
│   ├── interim/         # Intermediate processed data
│   └── processed/       # Final processed data ready for modeling
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code package
│   └── movie_recommender/
│       ├── data/        # Data loading and processing
│       ├── features/    # Feature engineering
│       ├── models/      # Model training and prediction
│       ├── visualization/ # Visualization utilities
│       └── utils/       # Utility functions
├── pipelines/           # Pipeline scripts
├── tests/               # Unit and integration tests
├── artifacts/           # Model outputs and visualizations
│   ├── models/          # Trained models
│   ├── figures/         # Generated figures
│   └── metrics/         # Model metrics
└── docs/                # Additional documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- Git
- DVC (Data Version Control)

### Installation

1. Clone the repository
2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate movie-recommender
```

3. Initialize DVC (if not already done):

```bash
dvc init
```

### Running the Pipeline

To run the complete pipeline:

```bash
dvc repro
```

To run a specific stage:

```bash
dvc repro <stage_name>
```

### Running the Optimized Pipelines

For convenience, two scripts have been added to run the optimized pipelines directly:

#### Feature Engineering

The feature engineering pipeline processes raw movie and user data to create features for the recommendation model:

```bash
python run_feature_engineering.py
```

This script:

1. Loads the merged dataset from `data/interim/final_merged_data.parquet` (or .csv)
2. Processes movie features, user profiles, and interaction features using parallel processing
3. Saves the processed features to `data/processed/`

#### Model Training

After feature engineering, you can train the recommendation model:

```bash
python run_training_pipeline.py --model_type neuro_fuzzy
```

Options:

- `--features_dir`: Directory containing processed features (default: `data/processed`)
- `--output_dir`: Directory to save trained models (default: `models`)
- `--model_type`: Type of model to train (`neuro_fuzzy` or `embedding_neuro_fuzzy`)

## Development

The `Makefile` provides common commands for development:

```bash
make help           # Show available commands
make setup          # Set up the development environment
make test           # Run tests
make lint           # Run linting
make clean          # Clean build artifacts
```

## License

[Specify your license here]

## Acknowledgments

- [List any acknowledgments, datasets, or references used]
