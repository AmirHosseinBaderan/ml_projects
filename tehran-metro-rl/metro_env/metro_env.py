import math

def haversine(lat1,lon1,lat2,lon2):
    R = 6371
    
    lat1,lon1,lat2,lon2 = map(math.radians,[lat1,lon1,lat2,lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

class MetroEnv:
    def __init__(self,stations,graph):
        self.stations = stations
        self.graph = graph
        
    def reset(self,start,goal):
        self.state = start
        self.goal = goal
        
        return self.state
    
    def actions(self,state):
        return self.graph[state]
    
    def step(self,action):
        if action not in self.graph[self.state]:
            return self.state, -10, False
        
        prev = self.state
        self.state = action
        
        if self.state == self.goal:
            return self.state,100,True
        
        prev_line = self.stations[prev].get("line",[])
        next_line = self.stations[self.state].get("line",[])
        
        line_switch_penalty = 0
        if not set(prev_line).intersection(set(next_line)):
            line_switch_penalty = -5
        
        reward = -1
        try:
            s1 = self.stations[prev]
            s2 = self.stations[action]
            
            if s1.get('latitude') and s2.get('latitude'):
                dist = haversine(
                    float(s1["latitude"]), float(s1["longitude"]),
                    float(s2["latitude"]), float(s2["longitude"])
                )
                reward -= dist * 0.01
        except:
            pass
        
        reward += line_switch_penalty
        return self.state,reward,False
                  
    def get_line(self,station):
        return self.stations[station].get("line",[])