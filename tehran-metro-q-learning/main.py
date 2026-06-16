import random
from graph_utils import get_neighbors,build_graph
from metro_env import TehranMetroEnv
import numpy as np

graph = build_graph()
env = TehranMetroEnv(graph)
stations = list(graph.keys())
station_to_id = {s:i for i, s in enumerate(stations)}
id_to_station = {i:s for s,i in station_to_id.items()}
Q = np.zeros((len(stations),len(stations)))

#Hyperparameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor 
epsilon = 1.0 # exploration
epsilon_decay = 0.995
epsilon_min = 0.05

def choise_action(state_id):
    if random.random() < epsilon:
        return random.choice(range(len(stations)))
    else:
        return np.argmax(Q[state_id])


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

# Training loop
for episode in range(1000):
    start = random.choice(stations)
    goal = random.choice(stations)
    
    state = env.reset(start,goal)
    state_id = station_to_id[state]
    
    done = False
    step = 0
    
    while not done and step < 100:
        action_id = choise_action(state_id)
        action = id_to_station[action_id]
        
        next_state,reward,done = env.step(action)
        next_state_id = station_to_id[next_state]
        
        # Q-Learning update
        Q[state_id][action_id] = Q[state_id][action_id] + alpha * (
            reward + gamma * np.max(Q[next_state_id]) - Q[state_id][action_id]
        )
        
        state_id = next_state_id
        step += 1
        
    # decay epsilon
    epsilon = max(epsilon_min,epsilon * epsilon_decay)