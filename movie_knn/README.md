# Movie KNN from Scratch

## What it does
A content-based movie recommender implemented from scratch using KNN and genre features. Recommends movies similar to a given title based on genre vectors and popularity.

## How it works
1. Loads ratings and movie metadata from tab-separated files
2. Computes per-movie statistics: number of ratings and average rating
3. Normalizes the rating count to `[0, 1]` for popularity comparison
4. Builds a dictionary mapping `movie_id` to `[name, genre_vector, normalized_count, mean_rating]`
5. For a given movie, computes distance to every other movie using:
   - Cosine distance between 20-dimensional genre one-hot vectors
   - Absolute difference in normalized popularity
6. Sorts by combined distance and returns the top `k` nearest neighbors
7. Computes the average mean rating of those neighbors

## Implementation
- `main.py`:
  - Loads `movies_data.txt` (ratings) and `movies_item.txt` (movie metadata)
  - Groups ratings to get `size` and `mean` rating per movie
  - Normalizes rating count using min-max scaling
  - `compute_distance(a, b)` — returns `(genre_cosine_distance, popularity_abs_distance)`
  - `get_neighbors(movie_id, k)` — brute-force search over all movies, sorted by tuple distance
  - Prints top 10 neighbors and their average rating

## Run
```bash
python main.py
```
