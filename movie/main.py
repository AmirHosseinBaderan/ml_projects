from engine import build_engine
from data import load_data

df = load_data()

recommender,search_engine = build_engine(df)

while True:

    cmd = input("\n1-Search\n2-Recommend\n0-Exit: ")

    if cmd == "0":
        break

    elif cmd == "1":
        q = input("Search: ")
        results = search_engine.search(q)

        if len(results) == 0:
            print("No movies found")
            continue
        
        for i, movie in enumerate(results.index, start=1):
            row = results.loc[movie]
            print(f"{i}. {movie} (rating={row['mean']:.2f}, votes={row['count']})")
    elif cmd == "2":
        movie = input("Movie: ")
        print(recommender.recommend(movie))