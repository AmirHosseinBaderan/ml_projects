import gym
import numpy as np
import random

env = gym.make('Taxi-v3',render_mode='ansi')
state,info = env.reset()

taxi = env.unwrapped
init_state = taxi.encode(2,3,2,0)
taxi.s = init_state

print(env.render())

q_table = np.zeros([env.observation_space.n,env.action_space.n])

epochs = 10000
exploration = 0.1
learning_rate = 0.1
discount_factor = 0.6

for tax_run in range(epochs):
    state,info = env.reset()
    done = False
    
    while not done:
        random_value = random.uniform(0,1)
        if random_value < exploration:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state,reward,done,info = env.step(action)
        prev_q = q_table[state,action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)
        q_table[state,action] = new_q
        state = next_state