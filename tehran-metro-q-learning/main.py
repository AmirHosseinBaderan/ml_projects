import json

with open('./data/stations.json') as f:
    stations = json.load(f)

graph = {}
for station_name,station_data in stations.items():
    graph[station_name] = station_data["relations"]

# graph validation
invalid_relations = []
for station_name,neighbors in graph.items():
    for neighbor in neighbors:
        if neighbor not in graph:
            invalid_relations.append(
                (station_name,neighbor)
            )
            
print(f"invalid relations : {len(invalid_relations)}")
for item in invalid_relations[:10]:
    print(item)