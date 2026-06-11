def build_movie_matrix(df):
    return df.pivot_table(
        index='user_id',
        columns='title',
        values='rating'
    )