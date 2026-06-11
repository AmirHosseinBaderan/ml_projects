import pandas as pd

ratings = pd.read_csv("./data/u.data",sep="\t",names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('./data/u.item',sep='|',encoding='latin-1',header=None,usecols=[0,1],names=['movie_id','title'])
users = pd.read_csv('./data/u.user',sep='|',header=None,names=['user_id','age','gender','occupation','zip_code'])

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
    corr_df = pd.DataFrame(similar_movies,columns=['correlation'])
    
    corr_df.dropna(inplace=True)
    recommendatoions = corr_df.join(
        movie_stats['count']
    )
    
    recommendatoions = recommendatoions[recommendatoions['count'] > 100]
    recommendatoions = recommendatoions.sort_values(
        'correlation',
        ascending=False
    )
    
    return recommendatoions.head(10)

movie = input('movie name : ')
recommendations = recommend(movie_name=movie)
print(recommendations)