import random

def choose_action(state,q_table,epsilon):
    actions = q_table[state]
    
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))
    
    return max(actions,key=actions.get)