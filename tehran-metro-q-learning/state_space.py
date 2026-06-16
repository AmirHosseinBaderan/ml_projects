def build_state_space(stations):
    states = []
    
    for current_station in stations:
        for goal_station in stations:
            states.append(
                (
                    current_station,
                    goal_station
                )
            )
    return states

def build_state_mappings(states):
    state_to_id = {
        state:idx
        for idx,state in enumerate(states)
    }
    
    id_to_state = {
        idx:state
        for idx,state in enumerate(states)
    }
    
    return state_to_id,id_to_state