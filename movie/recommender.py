import pandas as pd

class MovieRecommender:
    def __init__(self,model,movie_features,movie_stats):
        self.model = model
        self.movie_features = movie_features
        self.movie_stats = movie_stats
        
    def recommend(self,movie_name,top_n=10,min_votes=100):
        vector = self.movie_features.loc[movie_name].values.reshape(1,-1)
        distance ,indicaes = self.model.kleighbors(
            vector,
            n_neighbors=top_n+1
        )
        
        movies = self.movie_features.index[indicaes[0]]
        
        recs = pd.DataFrame({
            'title':movies,
            'similarity':1 - distance[0]
        }).set_index('title')
        
        recs = recs.join(self.movie_stats)
        recs = recs.drop(movie_name,errors='ignore')
        recs = recs[
            recs['count'] >= min_votes
        ]
        
        recs['score'] = (
            recs['similarity'] * 0.7 +
            recs['norm_count'] * 0.2 +
            recs['mean'] * 0.1
        )
        
        result = recs.sort_values(
            'score',
            ascending=False
        ).head(top_n)
        
        return result