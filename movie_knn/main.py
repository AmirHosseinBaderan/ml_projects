import pandas as pd
import numpy as np
from scipy import spatial
import operator

# read data
r_cols = ['user_id','movie_id','rating']
ratings = pd.read_csv('./data/movies_data.txt',sep='\t',names=r_cols,usecols=range(3))

# grouping items
movie_properties = ratings.groupby('movie_id').agg({'rating':[np.size,'mean']})

movie_num_ratings = pd.DataFrame(movie_properties['rating']['size'])
movie_normalize_num_ratings = movie_num_ratings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

movies_dictionary = {}

with open('./data/movies_item.txt') as f:
    for line in f:
        fileds = line.rstrip('\n').split('|')
        movieId = int(fileds[0])
        name = fileds[1]
        geners = [int(x) for x in fileds[5:25]]
        movies_dictionary[movieId] = [
            name,
            geners,
            movie_normalize_num_ratings.loc[movieId].get('size'),
            movie_properties.loc[movieId].rating.get('mean')
        ]
        
def compute_distance(a,b):
    genres_a = a[1]
    genres_b = b[1]
    
    genre_distance = spatial.distance.cosine(genres_a,genres_b)
    popularity_a = a[2]
    popularity_b = b[2]
    
    popularity_distance = abs(popularity_a - popularity_b)
    
    return genre_distance,popularity_distance

def get_neighbors(movie_id,k):
    distances = []
    for movie in movies_dictionary:
        if movie != movie_id:
            dist = compute_distance(movies_dictionary[movie],movies_dictionary[movie_id])
            distances.append((movie,dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

k = 10
average_rating = 0

neighbors = get_neighbors(1,k)
for neighbor in neighbors:
    average_rating += movies_dictionary[neighbor][3]
    print(f'{movies_dictionary[neighbor][0]} : {movies_dictionary[neighbor][3]}')
    
average_rating /= float(k)
print('------ average rating -------')
print(f"average : {average_rating}")