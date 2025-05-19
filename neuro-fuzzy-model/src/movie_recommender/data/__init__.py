"""
Movie recommender data processing package.

This package contains functions for processing and merging movie recommendation data,
including ratings, tags, and other metadata.
"""

# Data processing module initialization


from .merge_user_rate import (
    merge_ratings_movies,
    merge_with_links
)


# Add version info
__version__ = '0.1.0'
