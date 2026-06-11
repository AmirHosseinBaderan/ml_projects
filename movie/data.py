import pandas as pd

def get_ratings():
    return pd.read_csv(
        "./data/u.data",
        sep="\t",
        names=['user_id','movie_id','rating','timestamp']
    )
    
def get_movies():
    return pd.read_csv(
        './data/u.item',
        sep='|',
        encoding='latin-1',
        header=None,
        usecols=[0,1],
        names=['movie_id','title']
    )
    
def get_users():
    return pd.read_csv(
        './data/u.user',
        sep='|',
        header=None,
        names=['user_id','age','gender','occupation','zip_code']
    )
    
def load_data():
    ratings = get_ratings()
    movies = get_movies()
    users = get_users()

    df = pd.merge(ratings, movies, on="movie_id")
    return df