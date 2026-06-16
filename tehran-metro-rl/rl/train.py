def train(env,agent,start,goal,episodes=500):
    for ep in range(episodes):
        state = env.reset(start,goal)
        done = False
        steps = 0
        
        while not done and steps < 100:
            actions = env.actions(state)
            action = agent.choose_action(state,actions)
            
            next_state,reward,done = env.step(action)
            next_actions = env.actions(next_state)
            
            agent.learn(state,action,reward,next_state,done,next_actions)
            
            state = next_state
            steps += 1
            
        if ep % 50 == 0:
            print(f'Episode {ep}')