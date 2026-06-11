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

# test 
toy_story = movie_matrix['Toy Story (1995)']

similar_movies = movie_matrix.corrwith(toy_story)

corr_df = pd.DataFrame(similar_movies,columns=['correlation'])
corr_df.dropna(inplace=True)

# quality filter
movie_stats = df.groupby('title')['rating'].agg(['mean','count'])

recommendations = corr_df.join(movie_stats['count'])
# filter recommendations
recommendations = recommendations[recommendations['count'] > 100]

rec_items = recommendations.sort_values('correlation',ascending=False).head(10)
print(rec_items)