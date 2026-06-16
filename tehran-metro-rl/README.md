# Tehran Metro Route Planner (Reinforcement Learning)

A reinforcement learning-based route planner for Tehran's metro system that finds optimal paths between stations using Q-Learning.

## What It Does

This project implements an intelligent route planner for Tehran's metro network. Given a start station and a destination station, it uses a trained Q-Learning agent to find the best path through the metro graph. The agent learns to minimize travel cost while considering:

- **Distance-based penalties**: Longer physical distances between stations incur higher costs
- **Line-switching penalties**: Changing metro lines adds a penalty to discourage unnecessary transfers
- **Goal reward**: Reaching the destination provides a large positive reward

The result is an optimal or near-optimal route that balances shortest distance with minimal line changes.

## How It's Implemented

### Project Structure

```
tehran-metro-rl/
├── main.py                  # Entry point: user interaction and orchestration
├── data/
│   └── stations.json        # Tehran metro stations with coordinates, lines, and relations
├── metro_env/
│   └── metro_env.py         # Custom Gym-like environment for the metro network
├── rl/
│   ├── agent.py             # Q-Learning agent implementation
│   └── train.py             # Training loop
└── utils/
    ├── graph.py             # Graph construction from station data
    ├── graph_viz.py         # Graph visualization
    ├── learning_plot.py     # Training reward curve plotting
    └── visualize.py         # Path visualization
```

### Components

#### 1. Environment (`metro_env/metro_env.py`)

The `MetroEnv` class wraps the metro network as a reinforcement learning environment:

- **State**: Current station name
- **Action**: Adjacent station to move to (from the graph)
- **Reward**:
  - `+100` for reaching the goal
  - `-10` for invalid moves (not connected)
  - `-1` base cost per step
  - `-5` penalty for switching metro lines
  - Distance-based penalty proportional to Haversine distance between stations
- **Done**: True when the agent reaches the goal

The environment uses the **Haversine formula** to calculate real-world distances between stations using their latitude/longitude coordinates.

#### 2. Q-Learning Agent (`rl/agent.py`)

The `QLearningAgent` implements the classic Q-Learning algorithm with:

- **Q-table**: A `defaultdict` mapping `(state, action)` pairs to Q-values
- **Hyperparameters**:
  - `alpha = 0.1` — learning rate
  - `gamma = 0.9` — discount factor
  - `epsilon = 0.2` — exploration rate (epsilon-greedy policy)
- **Action selection**: Epsilon-greedy strategy (20% random exploration, 80% exploitation)
- **Update rule**: Standard Q-Learning Bellman equation

#### 3. Training (`rl/train.py`)

The training loop runs for a configurable number of episodes (default: 1000):

1. Reset environment to start station
2. At each step, agent chooses an action (epsilon-greedy)
3. Environment returns next state, reward, and done flag
4. Agent updates Q-value using the Bellman equation
5. Episode ends when goal is reached or max steps (100) exceeded
6. Reward history is recorded for analysis

#### 4. Graph & Data (`utils/graph.py`, `data/stations.json`)

- Stations are loaded from a JSON file containing:
  - Station names (English and Persian)
  - Metro line numbers
  - Geographic coordinates (latitude/longitude)
  - Adjacent station relations
- An adjacency graph is built where each station maps to its connected neighbors

#### 5. Path Extraction (`main.py`)

After training, the best path is extracted by greedily following the highest Q-value actions from start to goal.

## Algorithm: Q-Learning

**Q-Learning** is a model-free, off-policy reinforcement learning algorithm. It learns the optimal action-value function `Q(s, a)` by iteratively updating estimates based on experienced rewards.

### Q-Learning Update Equation

```
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `s` = current state (current station)
- `a` = action taken (next station)
- `r` = reward received
- `s'` = next state (resulting station)
- `α` = learning rate (how much new info overrides old)
- `γ` = discount factor (importance of future rewards)

### Why Q-Learning?

- **No environment model needed**: Learns directly from interaction
- **Off-policy**: Can learn optimal policy while exploring suboptimal actions
- **Simple and effective**: Works well for small-to-medium state spaces like metro networks
- **Converges to optimal policy**: Given enough episodes, finds the optimal route

## How It Works

1. **Load Data**: Parse `stations.json` to build the metro graph
2. **Initialize**: Create the environment and Q-Learning agent
3. **Train**: Run 1000 episodes where the agent explores the network, learning Q-values for each state-action pair
4. **Extract Path**: After training, follow the highest-valued actions from start to goal
5. **Visualize**: Display the learning curve and the resulting path on the metro graph

## Usage

```bash
cd tehran-metro-rl
python main.py
```

Then enter your start and destination stations when prompted.

## Requirements

- Python 3.x
- `networkx` — graph operations and visualization
- `matplotlib` — plotting learning curves and graph display
- `json` — data loading (standard library)
- `math` — Haversine distance calculation (standard library)
