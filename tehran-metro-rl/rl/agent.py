import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self,alpha=0.1,gamma=0.9,epsilon=0.2):
        self.q = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def choose_action(self,state,actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        return max(actions,key=lambda a:self.q[state][a])
    
    def learn(self,s,a,r,s2,done,next_actions):
        best_next = 0 if done else max(self.q[s2].values(),default=0)
        
        self.q[s][a] += self.alpha * (
            r + self.gamma * best_next - self.q[s][a]
        )