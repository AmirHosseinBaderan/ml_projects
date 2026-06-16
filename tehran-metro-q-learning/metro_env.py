from collections import deque

class TehranMetroEnv:
    def __init__(self,graph):
        self.graph = graph
        self.current = None
        self.goal = None
        self.last_state = None
        
    def reset(self,start,goal):
        self.current = start
        self.goal = goal
        self.last_state = None
        return (
            self.current,
            self.goal
        )
        
    def get_distance(self,start,goal):
        queue = deque([(start,0)])
        visited = set()
        
        while queue:
            node,dist = queue.popleft()
            if node == goal:
                return dist
            
            visited.add(node)
            for n in self.graph[node]:
                if n not in visited:
                    queue.append((n,dist+1))
        
        return 999
    
    def step(self,action):
       # invalid action (not neighbor)
        if action not in self.graph[self.current]:
          return (self.current,self.goal),-20,False
       
        prev_state = self.current
        # move
        self.current = action
        
        # goal reached 
        if self.current == self.goal:
            self.last_state = prev_state
            return (self.current,self.goal),100,True
        
        # loop penalty (backtracking)
        if self.current == self.last_state:
            reward = -10
        else:
            reward = -1
        
        # update memory 
        self.last_state = prev_state
        return (self.current,self.goal),reward,False
        
    
    def get_available_actions(self):
        return self.graph[self.current]