import gym

env = gym.make('Taxi-v3',render_mode='ansi')
state,info = env.reset()

print(env.render())