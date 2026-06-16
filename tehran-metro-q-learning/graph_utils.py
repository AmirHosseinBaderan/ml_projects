import json

def get_neighbors(graph,station):
    return graph.get(station,[])

def build_graph():
    with open('./data/stations.json') as f:
        stations = json.load(f)

    graph = {}
    for station_name,station_data in stations.items():
        graph[station_name] = station_data["relations"]
    
    return graph
