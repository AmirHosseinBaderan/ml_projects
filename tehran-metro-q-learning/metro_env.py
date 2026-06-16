class TehranMetroEnv:
    def __init__(self,graph):
        self.graph = graph
        self.current = None
        self.goal = None
        
    def reset(self,start,goal):
        self.current = start
        self.goal = goal
        return (
            self.current,
            self.goal
        )
    
    def step(self,action):
        # invalid move
        if action not in self.graph[self.current]:
            return (self.current,self.goal), -100,False
        
        self.current = action
        if self.current == self.goal:
            return (self.current,self.goal),100,True
        
        return (self.current,self.goal),-1,False