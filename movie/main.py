import pandas as pd
from data import get_movies,get_ratings,get_users

ratings = get_ratings()
movies = get_movies()
users = get_users()

# merge ragins + movives
df = pd.merge(ratings,movies,on='movie_id')


# movie-user matrix
movie_matrix = df.pivot_table(
    index='user_id',
    columns='title',
    values='rating'
)

movie_stats = df.groupby('title')['rating'].agg(
    ['mean', 'count']
)


def recommend(movie_name):
    movie_ratings = movie_matrix[movie_name]
    
    similar_movies = movie_matrix.corrwith(movie_ratings)
    
    corr_df = pd.DataFrame(
        similar_movies,
        columns=['correlation']
    )
    
    corr_df.dropna()
    recommendations = corr_df.join(
        movie_stats['count']
    )
    
    recommendations = recommendations[
        recommendations['count'] > 100
    ]
    
    recommendations = recommendations.sort_values(
        'correlation',
        ascending=False
    )
    
    recommendations = recommendations[
        recommendations.index != movie_name
    ]
    
    return recommendations.head(10)

movie = input('movie name : ')
recommendations = recommend(movie_name=movie)
print(recommendations)