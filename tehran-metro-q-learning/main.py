import json
import random
from graph_utils import get_neighbors,build_graph
from metro_env import TehranMetroEnv

graph = build_graph()
env = TehranMetroEnv(graph)

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

state = env.reset("Tajrish","Shahid Sadr")
print(state)
next_state,reward,done = env.step("Gheytariyeh")
print(next_state,reward,done)