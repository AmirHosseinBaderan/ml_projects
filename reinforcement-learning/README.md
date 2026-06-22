.# Taxi Reinforcement Learning

A Python implementation of Q-Learning algorithm to solve the classic Taxi-v3 environment from OpenAI Gymnasium.

## What It Does

This project trains an AI agent to solve the Taxi environment, where the goal is to pick up passengers and drop them off at their destinations efficiently. The agent learns through reinforcement learning to maximize cumulative rewards.

## Implementation

### Environment

- Uses `gymnasium` (formerly gym) with the `Taxi-v4` environment
- The taxi environment consists of:
  - A 5x5 grid with designated pickup and drop-off locations
  - A taxi that must navigate to pick up passengers and deliver them
  - Actions: move south, north, east, west, pickup, dropoff

### Q-Learning Algorithm

The implementation uses a Q-table approach with the following parameters:

- **Q-Table**: A table of shape `[states, actions]` initialized to zeros
- **Epochs**: 10,000 training episodes
- **Exploration Rate**: 0.1 (epsilon-greedy strategy)
- **Learning Rate**: 0.1
- **Discount Factor**: 0.6

### Training Process

1. Initialize Q-table with zeros
2. For each episode:
   - Reset environment to initial state
   - For each step:
     - Choose action: random (exploration) or best known (exploitation)
     - Take action and observe reward and next state
     - Update Q-value using the Q-learning formula:
       ```
       Q(s,a) = (1 - α) × Q(s,a) + α × (reward + γ × max(Q(s',a')))
       ```
     - Transition to next state
3. After training, run 10 demonstration trips using the learned policy

### Key Functions

- **`clear_terminal()`**: Clears the terminal for clean visualization
- **Training Loop**: Implements Q-learning update rule
- **Demonstration Loop**: Shows the trained agent in action with visual output

## Usage

```bash
cd reinforcement-learning
python main.py
```

## Dependencies

- `gymnasium`: For the Taxi environment
- `numpy`: For Q-table operations
- `random`: For exploration
- `time`: For visualization delays

## Example Output

The program displays:
1. Initial state of the environment
2. Training progress (silent)
3. 10 demonstration trips showing the taxi navigating to pick up and drop off passengers

## How It Works

The agent learns to:
1. Identify the passenger's location
2. Navigate to the passenger
3. Pick up the passenger
4. Navigate to the destination
5. Drop off the passenger

The Q-table stores the expected future rewards for each state-action pair, allowing the agent to make optimal decisions through experience.