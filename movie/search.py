

def search(movie_stats,query,limit=10):
    q = query.lower()
    
    results = movie_stats.loc[
        movie_stats.index.str.lower().str.contains(q)
    ]
    
    return results.sort_values(
        'count',
        ascending=False
    )