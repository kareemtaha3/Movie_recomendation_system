# Optimized Movie Feature Engineering

This module provides optimized feature engineering for movie recommendation systems with GPU acceleration support and memory-efficient batch processing.

## Key Improvements

1. **GPU Acceleration**: Uses PyTorch for tensor operations when a GPU is available
2. **Memory-Efficient Batch Processing**: Processes data in smaller batches to prevent memory errors
3. **Robust Error Handling**: Handles missing values, division by zero, and other edge cases
4. **Progress Tracking**: Shows progress bars during processing
5. **Automatic Fallback**: Falls back to CPU processing if GPU processing fails

## Usage

### Basic Usage (CPU)

```python
from movie_recommender.features.movie_feature_engineering import engineer_movie_features

# Process with CPU (default)
engineered_df = engineer_movie_features(
    your_dataframe,
    batch_size=5000,  # Adjust based on your available memory
    use_gpu=False     # Default is False
)
```

### GPU Acceleration

To use GPU acceleration, set the environment variable before importing:

```python
import os
os.environ['USE_GPU'] = 'True'  # Set this before importing the module

from movie_recommender.features.movie_feature_engineering import engineer_movie_features

# Process with GPU
engineered_df = engineer_movie_features(
    your_dataframe,
    batch_size=5000,
    use_gpu=True
)
```

## Performance Tips

1. **Batch Size**: Adjust the `batch_size` parameter based on your available memory. Smaller batches use less memory but may take longer to process.

2. **GPU Memory**: If you encounter GPU memory errors, try reducing the batch size further.

3. **Monitoring**: The code includes logging to monitor progress and performance. Check the logs for insights.

4. **Pre-processing**: Consider cleaning your data (handling missing values, etc.) before passing it to the feature engineering function for better performance.

## Requirements

- pandas
- numpy
- scikit-learn
- tqdm (for progress bars)
- PyTorch (optional, for GPU acceleration)
