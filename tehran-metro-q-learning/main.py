import random
from graph_utils import get_neighbors,build_graph
from state_space import build_state_space,build_state_mappings
from metro_env import TehranMetroEnv
import numpy as np
from q_table import create_q_table
from policy import choose_action

graph = build_graph()
env = TehranMetroEnv(graph)
stations = list(graph.keys())

#Hyperparameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor 
epsilon = 1.0 # exploration
epsilon_decay = 0.995
epsilon_min = 0.05

# states
states = build_state_space(stations)
state_to_id, id_to_state = build_state_mappings(states)
q_table = create_q_table(states,graph)

# Training 
for episode in range(20000):
    start = random.choice(stations)
    goal = random.choice(stations)
    
    state = env.reset(start,goal)
    
    done = False
    step =0
    
    while not done and step < 100:
        # choose action 
        action = choose_action(state,q_table,epsilon)
        
        # step in env 
        next_state,reward,done = env.step(action)
        
        # safety check 
        if next_state not in q_table:
            break
        
        # Q- values
        old_value = q_table[state][action]
        next_max = max(q_table[next_state].values())

        # Q update
        q_table[state][action] = old_value + alpha *(
            reward + gamma * next_max - old_value
        )
        
        # move forward
        state = next_state
        step += 1
        
    # exploration decay
    epsilon = max(epsilon_min,epsilon * epsilon_decay)
    
    if epsilon % 100 == 0:
        print(f"Episode {episode} done, Epsilon={epsilon}")
        
state = env.reset("Tajrish", "Meydan-e Azadi")
for i in range(30):
    action = choose_action(state,q_table,0.0) 
    next_state,reward,done = env.step(action)
    
    print(state, "->", action, "->", next_state)
    
    state = next_state
    if done:
        print("GOAL FOUND :)")
        break