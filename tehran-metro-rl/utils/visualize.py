

def print_policy(graph,agent):
    print("Learned Policy (Bestaction per station)")
    
    for state in graph:
        actions = graph[state]
        
        if not actions:
            continue
        
        best_actoin = max(actions,key=lambda a: agent.q[state][a])
        print(f"{state} -> {best_actoin}")