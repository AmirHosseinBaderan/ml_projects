class MovieSearchEngine:

    def __init__(self, movie_stats):
        self.movie_stats = movie_stats

    def search(self, query, limit=10):

        q = query.lower()

        results = self.movie_stats.loc[
            self.movie_stats.index.str.lower().str.contains(q)
        ]

        results = results.sort_values(
            'count',
            ascending=False
        )

        return results.head(limit)