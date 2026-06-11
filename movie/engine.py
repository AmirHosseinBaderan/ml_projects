import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommender import MovieRecommender

def build_engine(df):
    movie_matrix = build_movie_matrix(df)
    movie_features = movie_matrix.fillna(0).T
    
    similarity = cosine_similarity(movie_features)
    
    similarity_df = pd.DataFrame(
        similarity,
        index=movie_features.index,
        columns=movie_features.index
    )
    
    movie_stats = df.groupby('title')['rating'].agg(['mean','count'])
    movie_stats['norm_count'] = (
        movie_stats['count'] - movie_stats['count'].min()
    ) / (
        movie_stats['count'].max() - movie_stats['count'].min()
    )
    
    recomender = MovieRecommender(similarity_df,movie_stats)
    
    return recomender,movie_stats
    
    
def build_movie_matrix(df):

    return df.pivot_table(
        index='user_id',
        columns='title',
        values='rating'
    )