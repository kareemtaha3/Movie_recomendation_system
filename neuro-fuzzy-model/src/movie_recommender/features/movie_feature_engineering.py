import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple

def engineer_movie_features(
    df: pd.DataFrame,
    top_n_companies: int = 10,
    top_n_actors: int = 5
) -> pd.DataFrame:
    """
    Engineer movie/item features for deep learning models.

    Parameters:
    - df: Input DataFrame with columns as provided.
    - top_n_companies: Number of top production companies to encode.
    - top_n_actors: Number of top actors to consider for popularity.

    Returns:
    - DataFrame with new feature columns appended.
    """
    df = df.copy()
    
    # 1. Release year
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year

    # 2. Runtime normalization
    df['runtime_norm'] = (df['runtime'] - df['runtime'].mean()) / df['runtime'].std()

    # 3. Log-scale budget & revenue
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])

    # 4. Normalized popularity & vote_average
    df['popularity_norm'] = (
        df['popularity'] - df['popularity'].min()
    ) / (df['popularity'].max() - df['popularity'].min())
    df['vote_average_norm'] = (
        df['vote_average'] - df['vote_average'].min()
    ) / (df['vote_average'].max() - df['vote_average'].min())

    # 5. Log vote count
    df['log_vote_count'] = np.log1p(df['vote_count'])

    # 6. Language code
    df['language_code'] = df['original_language'].astype('category').cat.codes

    # 7. Genres multi-hot
    df['genres_list'] = df['genres'].str.split('|')
    mlb_genres = MultiLabelBinarizer()
    genres_mh = mlb_genres.fit_transform(df['genres_list'])
    genres_df = pd.DataFrame(
        genres_mh,
        columns=[f'genre_{g}' for g in mlb_genres.classes_],
        index=df.index
    )

    # 8. Production companies encoding (top N)
    df['companies_list'] = df['production_companies'].str.split('|')
    # Find top N companies
    all_companies = [
        comp for sub in df['companies_list'] if isinstance(sub, list) for comp in sub
    ]
    top_companies = pd.Series(all_companies).value_counts().head(top_n_companies).index
    df['company_code'] = df['companies_list'].apply(
        lambda comps: next((i for i, c in enumerate(top_companies) if c in comps), top_n_companies)
        if isinstance(comps, list) else top_n_companies
    )

    # 9. Cast list and popularity
    df['cast_list'] = df['cast'].str.split('|')
    # Compute actor popularity (mean popularity of movies they appear in)
    actor_pop = (
        df[['cast_list', 'popularity']]
        .explode('cast_list')
        .groupby('cast_list')['popularity']
        .mean()
    )
    df['cast_popularity'] = df['cast_list'].apply(
        lambda lst: np.mean([actor_pop.get(actor, 0) for actor in lst]) if isinstance(lst, list) else 0
    )

    # 10. Director features
    df['director_name'] = df['crew']  # Already processed to just director name
    df['director_code'] = df['director_name'].astype('category').cat.codes
    # Director popularity: total vote_count across their movies
    dir_pop = df.groupby('director_name')['vote_count'].sum()
    df['director_popularity'] = df['director_name'].map(dir_pop)

    # Combine all engineered features
    engineered = pd.concat(
        [df, genres_df],
        axis=1
    )

    return engineered

# Example usage:
# engineered_df = engineer_movie_features(your_dataframe)
