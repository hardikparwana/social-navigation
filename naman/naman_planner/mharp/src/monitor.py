

class Monitor(object): 
    def __init__(self, abstract_states):
        self.abstract_states = abstract_states
        self.timed_occupancy = {}
        self.pending_execution = {}
        self.cache = { }
        self.updated = { } 
        for abstract_state in self.abstract_states: 
            self.cache[abstract_state.id] = -1 
            self.updated[abstract_state.id] = True

    def register(self,robot_id,high_level_plan):
        for i,abstract_state in enumerate(high_level_plan): 
            self.timed_occupancy[abstract_state.id] =  {robot_id: i}
            self.updated[abstract_state.id] = True
        self.pending_execution[robot_id] = high_level_plan
    
    def get_robot_dynamic_cost(self,abstract_state, delta_t): 
        c = 0
        if self.updated[abstract_state.id]:
            if abstract_state.id in self.timed_occupancy:
                for robot in self.timed_occupancy[abstract_state.id]: 
                    if self.timed_occupancy[abstract_state.id][robot] in [delta_t, delta_t -1, delta_t + 1]: 
                        c += 1
            else:
                c += 0
            self.cache[abstract_state.id] = c
            # self.updated[abstract_state.id] = False
        else: 
            c = self.cache[abstract_state.id]
        return c
    
    def go_to_next_state(self, robot_id):
        self.pending_execution[robot_id].pop(0)
        for abstract_state in self.pending_execution[robot_id]: 
            self.timed_occupancy[abstract_state.id][robot_id] -= 1
            self.updated[abstract_state.id] = True
