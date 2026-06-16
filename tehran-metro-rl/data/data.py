import json

def load_stations(path="./stations.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)