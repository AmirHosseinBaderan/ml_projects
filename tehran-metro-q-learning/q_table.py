def create_q_table(states,graph):
    q_table = {}
    
    for state in states:
        current_station ,goal_station = state
        actions = graph[current_station]
        q_table[state] = {
            action:0.0
            for action in actions
        }
        
    return q_table