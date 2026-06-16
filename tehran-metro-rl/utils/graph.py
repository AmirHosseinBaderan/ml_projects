import json 

def load_stations(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)
    
def build_graph(stations):
    return {
        name:data.get("relations",[])
        for name,data in stations.items()
    }