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
    query = query.lower()
    matches = movie_stats.loc[
        movie_stats.index.str.lower().str.contains(query)
    ]
    
    matches = matches.sort_values(
        'count',
        ascending=False
    )
    
    return matches.head(limit)


while True:
    cmd = input("\n 1-Search \n 2-Recommendation \n 0-Exit \n\n Select : ")
    
    if cmd == "0":
        break
    elif cmd == "1":
        query = input("Search : ")
        results = search_movie(query)
        
        if len(results) == 0:
            print("No movies found")
            continue
        
        for i,movie in enumerate(results.index,start=1):
            row = results.loc[movie]
            
            print(
                f"{i}. {movie}"
                f"(rating={row['mean']:.2f}, votes={row['count']})"
            )
            
        choice = input(
            "\n Select movie number for recommdations (Enter to skip): "
        )
        if choice:
            selected_movie = results.index[int(choice) -1]
            print(f"\n Recommendations for : {selected_movie}\n")
            
            print(recommend(selected_movie))
            
    elif cmd == '2':
        movie = input("Movie name : ")
        print(
            recommend(movie)
        )
        
    else:
        print("Invalid command")
