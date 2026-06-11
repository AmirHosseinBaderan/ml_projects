import pandas as pd

ratings = pd.read_csv("./data/u.data",sep="\t",names=['user_id','movie_id','rating','timestamp'])

print(ratings.head())