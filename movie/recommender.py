
class MovieRecommnder:
    def __init__(self,similarity_df,movie_stats):
        self.similarity_df = similarity_df
        self.movie_stats = movie_stats
        
    def recommend(self,movie_name,top_n=10,min_votes=100):
        similar = self.similarity_df.loc[movie_name]
        
        recs = similar.to_frame('similarity')
        recs = recs.join(self.movie_stats)
        
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