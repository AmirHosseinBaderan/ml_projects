import pandas as pd
from data import get_movies,get_ratings,get_users
from sklearn.metrics.pairwise import cosine_similarity

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

movie_features = movie_matrix.fillna(0).T

cosine_sim = cosine_similarity(movie_features)
similarity_df = pd.DataFrame(
    cosine_sim,
    index=movie_features.index,
    columns=movie_features.index
)

movie_stats = df.groupby('title')['rating'].agg(
    ['mean', 'count']
)

def recommend(movie_name,top_n=10):
    similar = similarity_df[movie_name]
    similar = similar.sort_values(
        ascending=False
    )
    
    similar = similar.drop(movie_name)
    return similar.head(top_n)

def search_movie(query,limit=10):
    movie_titles = movies['title'].unique()
    
    query = query.lower()
    results = [
        title
        for title in movie_titles
        if query in title.lower()
    ]
    
    return results[:limit]

cmd = input('command : 1-Search , 2-Recommendation : ')
if cmd == '1':
    query = input('Input search : ')
    search_res = search_movie(query)
    print(search_res)
elif cmd == '2':
    movie = input('movie name : ')
    recommendations = recommend(movie_name=movie)
    print(recommendations)    

