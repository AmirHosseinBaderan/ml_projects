# Movie Recommender

## What it does
An interactive movie recommendation and search system. Uses collaborative filtering (item-based KNN) to recommend movies similar to a given title, and provides fuzzy title search over the MovieLens dataset.

## How it works
1. Loads MovieLens data (`u.data` ratings, `u.item` movies, `u.user` users) via `data.py`
2. Builds a user-by-movie rating matrix (`engine.py` / `feature_builder.py`)
3. Trains a `NearestNeighbors` model with cosine similarity on the movie-feature vectors
4. When recommending, finds nearest movie vectors, filters by minimum votes, and ranks by a weighted score combining similarity, popularity, and average rating
5. Search performs case-insensitive substring matching on movie titles sorted by popularity

## Implementation
- `data.py` — loads and merges MovieLens dataset files into a single DataFrame
- `feature_builder.py` — pivots ratings into a user × movie matrix (duplicate of `engine.build_movie_matrix`, not used directly)
- `engine.py` — orchestrates model training, movie statistics (mean rating, count, normalized count), and builds both the recommender and search engine
- `recommender.py:9` — `MovieRecommender.recommend()` queries the KNN model, joins with stats, applies a scoring formula:
  - `score = 0.7 * similarity + 0.2 * norm_count + 0.1 * mean`
  - Filters out the input movie and titles with fewer than `min_votes` ratings
- `search.py` — case-insensitive `str.contains` search over title index, sorted by rating count

> Note: `recommender.py` contains a bug on line 11 — `self.model.kleighbors` should be `self.model.kneighbors`.

## Run
```bash
python main.py
```
