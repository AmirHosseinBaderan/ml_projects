import pandas as pd
from recommender import MovieRecommender
from sklearn.neighbors import NearestNeighbors
from search import MovieSearchEngine

def build_engine(df):
    movie_matrix = build_movie_matrix(df)
    movie_features = movie_matrix.fillna(0).T
    
    model = NearestNeighbors(
        metric='cosine',
        algorithm='brute'
    )
    
    model.fit(movie_features)
    movie_stats = df.groupby('title')['rating'].agg({'mean','count'})
    movie_stats['norm_count'] = (
        movie_stats['count'] - movie_stats['count'].min()
    ) / (
        movie_stats['count'].max()- movie_stats['count'].min()
    )
    
    recommender = MovieRecommender(
        model,
        movie_features,
        movie_stats,
    )
    
    search_engine = MovieSearchEngine(movie_stats)
    
    return recommender,search_engine
    
    
def build_movie_matrix(df):

    return df.pivot_table(
        index='user_id',
        columns='title',
        values='rating'
    )