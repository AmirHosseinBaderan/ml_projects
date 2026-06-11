# Movie Recommendation System

A movie search and recommendation engine built using collaborative filtering with k-nearest neighbors (KNN).

## Project Overview

This system provides two main functionalities:
1. **Movie Search** - Find movies by title keyword
2. **Movie Recommendations** - Get similar movie suggestions based on user rating patterns

## How It Works

The system uses collaborative filtering to recommend movies:

1. **Data Processing**: Loads movie ratings data and transforms it into a user-movie rating matrix
2. **Feature Building**: Each movie becomes a feature vector based on ratings from all users
3. **Similarity Search**: Uses scikit-learn's `NearestNeighbors` with cosine distance to find similar movies
4. **Recommendation Scoring**: Combines similarity scores with popularity metrics to rank recommendations

## File Structure

| File | Description |
|------|-------------|
| `main.py` | CLI entry point - provides interactive menu for search and recommendations |
| `data.py` | Data loading utilities for MovieLens dataset files (u.data, u.item, u.user) |
| `feature_builder.py` | Builds the user-movie rating matrix (also duplicated in engine.py) |
| `engine.py` | Core engine that initializes the KNN model and creates recommender/search engine instances |
| `recommender.py` | `MovieRecommender` class - finds similar movies using KNN and ranks by composite score |
| `search.py` | `MovieSearchEngine` class - searches movies by title substring |

## Recommendation Algorithm

The recommendation score combines three factors:
- **Similarity (70%)** - 1 minus cosine distance to the target movie
- **Popularity (20%)** - Normalized vote count
- **Average Rating (10%)** - Mean rating of the movie

Movies must have at least 100 votes to be included in recommendations.

## Data Files

Located in `data/` directory:
- `u.data` - User ratings (100,000 records): user_id, movie_id, rating, timestamp
- `u.item` - Movie metadata (1,682 movies): movie_id, title, release_date, IMDb URL, genres
- `u.user` - User demographics (943 users): user_id, age, gender, occupation, zip_code

## Usage

```bash
python main.py
```

Options:
1. Search - Enter a movie title keyword to find movies
2. Recommend - Enter a movie name to get similar recommendations
0. Exit - Quit the program