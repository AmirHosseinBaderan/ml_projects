import gym
import numpy as np

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
discount_rate = 0.6