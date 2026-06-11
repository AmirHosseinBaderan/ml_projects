from engine import build_engine
from data import load_data
from search import search

df = load_data()

recommender, movie_stats = build_engine(df)

while True:

    cmd = input("\n1-Search\n2-Recommend\n0-Exit: ")

    if cmd == "0":
        break

    elif cmd == "1":
        q = input("Search: ")
        print(search(movie_stats, q))

    elif cmd == "2":
        movie = input("Movie: ")
        print(recommender.recommend(movie))