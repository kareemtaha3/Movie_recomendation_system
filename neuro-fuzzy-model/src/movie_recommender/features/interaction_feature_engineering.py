import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def engineer_interaction_features(
    ratings: pd.DataFrame,
    user_profiles: pd.DataFrame,
    movie_features: pd.DataFrame,
    genre_cols: list,
    keyword_vec_col: str = 'keywords_vector'
) -> pd.DataFrame:
    """
    Compute user-item interaction features for each rating record.

    Parameters:
    - ratings: DataFrame with columns ['userId', 'movieId', ...].
    - user_profiles: DataFrame indexed by userId containing:
        * user_genre_pref_<GENRE> for each genre in genre_cols
        * user_top_directors: list of director names
        * user_top_actors: list of actor names
        * user_pref_languages: list of language codes
        * user_avg_year: numeric
        * user_avg_log_budget: numeric
        * user_avg_log_revenue: numeric
        * user_keyword_vector: array-like vector of same dim as movie keyword vectors
    - movie_features: DataFrame indexed by movieId containing:
        * release_year, log_budget, log_revenue, original_language
        * director_name, cast_list (list of names)
        * keywords_vector: array-like
        * genre multi-hot columns matching genre_cols
    - genre_cols: list of genre column names in movie_features (e.g. ['genre_Action', ...])
    - keyword_vec_col: column name for movie keyword embedding

    Returns:
    - DataFrame of same length as ratings with new interaction feature columns.
    """
    df = ratings.copy()

    # Merge in user_profile features
    df = df.join(user_profiles, on='userId', rsuffix='_user')

    # Merge in movie_features
    df = df.join(movie_features, on='movieId', rsuffix='_movie')

    # 1. genre_affinity: dot product user_genre_pref & movie_genres
    user_genre_cols = [f'user_genre_pref_{g.split("_",1)[1]}' for g in genre_cols]
    df['genre_affinity'] = df.apply(
        lambda row: np.dot(row[user_genre_cols].values, row[genre_cols].values),
        axis=1
    )

    # 2. director_match: 1 if movie director in user's top directors
    df['director_match'] = df.apply(
        lambda row: int(row['director_name'] in row['user_top_directors']),
        axis=1
    )

    # 3. actor_overlap: fraction of movie's cast in user's top actors
    def compute_actor_overlap(cast, top_actors):
        if not cast: return 0.0
        return len(set(cast) & set(top_actors)) / len(cast)

    df['actor_overlap'] = df.apply(
        lambda row: compute_actor_overlap(row['cast_list'], row['user_top_actors']),
        axis=1
    )

    # 4. language_match: 1 if movie original_language in user's preferred languages
    df['language_match'] = df.apply(
        lambda row: int(row['original_language'] in row['user_pref_languages']),
        axis=1
    )

    # 5. release_year_distance
    df['release_year_distance'] = np.abs(df['release_year'] - df['user_avg_year'])

    # 6. budget_distance
    df['budget_distance'] = np.abs(df['log_budget'] - df['user_avg_log_budget'])

    # 7. revenue_distance
    df['revenue_distance'] = np.abs(df['log_revenue'] - df['user_avg_log_revenue'])

    # 8. keyword_similarity: cosine similarity of keyword vectors
    # Ensure vectors are 2D for sklearn
    def cos_sim(u_vec, m_vec):
        if u_vec is None or m_vec is None: return 0.0
        return cosine_similarity([u_vec], [m_vec])[0,0]

    df['keyword_similarity'] = df.apply(
        lambda row: cos_sim(row['user_keyword_vector'], row[keyword_vec_col]),
        axis=1
    )

    # Select only interaction feature columns to return
    interaction_cols = [
        'genre_affinity', 'director_match', 'actor_overlap',
        'language_match', 'release_year_distance',
        'budget_distance', 'revenue_distance', 'keyword_similarity'
    ]
    return df[interaction_cols]

# Example usage:
# feature_df = engineer_interaction_features(
#     ratings_df,
#     user_profiles_df,
#     movie_features_df,
#     genre_cols=[col for col in movie_features_df.columns if col.startswith('genre_')],
#     keyword_vec_col='keywords_vector'
# )
