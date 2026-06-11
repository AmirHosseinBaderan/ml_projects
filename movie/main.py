import pandas as pd

ratings = pd.read_csv("./data/u.data",sep="\t",names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('./data/u.item',sep='|',encoding='latin-1',header=None,usecols=[0,1],names=['movie_id','title'])
users = pd.read_csv('./data/u.user',sep='|',header=None,names=['user_id','age','gender','occupation','zip_code'])

# merge ragins + movives
df = pd.merge(ratings,movies,on='movie_id')
print(df.head())