def train(env,agent,start,goal,episodes=500):
    reward_history = []
    
    for ep in range(episodes):
        state = env.reset(start,goal)
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 100:
            actions = env.actions(state)
            action = agent.choose_action(state,actions)
            
            next_state,reward,done = env.step(action)
            next_actions = env.actions(next_state)
            
            agent.learn(state,action,reward,next_state,done,next_actions)
            
            state = next_state
            steps += 1
            total_reward += reward

        reward_history.append(total_reward)

        if ep % 50 == 0:
            print(f'Episode {ep} | Reward {total_reward}')
            
    return reward_history