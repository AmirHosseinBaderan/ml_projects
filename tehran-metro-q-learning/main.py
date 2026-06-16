import json
import random
from graph_utils import get_neighbors,build_graph

graph = build_graph()

def get_random_action(graph,current_station):
    neighbors = graph[current_station]
    return random.choice(neighbors)   
    
def run_episode(
    graph,
    start_station,
    goal_station,
    max_steps=100
):
    current_station = start_station
    path = [current_station]
    for step in range(max_steps):
        if current_station == goal_station:
            return True,path
        
        current_station = get_random_action(
            graph,
            current_station
        )
        
        path.append(current_station)
    return False,path

success,path = run_episode(graph,"Tajrish","Teatr-e Shahr")
print(success)
print(path)