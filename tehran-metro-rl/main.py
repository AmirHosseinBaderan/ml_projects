from utils.graph import build_graph,load_stations
from rl.train import train
from rl.agent import QLearningAgent
from metro_env.metro_env import MetroEnv

def get_best_path(env:MetroEnv,agent:QLearningAgent,start,goal):
    state = start
    path = [state]
    visited = set()
    
    while state != goal:
        actions = env.actions(state)
        action = max(actions,key=lambda a:agent.q[state][a])
        
        if action in visited:
            break
        
        visited.add(action)
        state = action
        path.append(state)
        
    return path

if __name__ == "__main__":
    stations = load_stations("./data/stations.json")
    graph = build_graph(stations)

    env = MetroEnv(stations, graph)
    agent = QLearningAgent()

    print("\n Metro Route Planner (Q-Learning)\n")

    print("Available stations:")
    for name in stations.keys():
        print("-", name)

    start = input("\nEnter start station: ").strip()
    goal = input("Enter goal station: ").strip()

    if start not in stations or goal not in stations:
        print("\n Invalid station name!")
        exit()

    print("\n Training agent...")
    train(env, agent, start, goal, 1000)
    
    path = get_best_path(env, agent, start, goal)

    print("\n Best path:")
    print(" -> ".join(path))