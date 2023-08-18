import numpy as np
class State(object):
    def __init__(self,id,ll_state = None):
        self.id = id
        self.ll_state = ll_state

    def __hash__(self):
        return self.id

    def __eq__(self, o):
        return self.id == o.id

    def __lt__(self,o):
        return self.id < o.id

    def __gt__(self,o):
        return self.id > o.id

    def get_centroid(self):
        return self.ll_state

    def __str__(self):
        return str(self.id)

class Interface(State):
    def __init__(self,id,ll_state = None):
        super(Interface,self).__init__(id,ll_state)
 
    def __lt__(self,o):
        if self.id[0] < o.id[0]:
            return True
        elif self.id[0] > o.id[0]:
            return False
        else:
            return self.id[1] < o.id[1]

    def __hash__(self):
        return str(self.id)

    def __eq__(self,o):
        if self.id[0] == o.id[0] and self.id[1] == o.id[1]:
            return True
        elif self.id[0] == o.id[1] and self.id[1] == o.id[0]:
            return True
        else:
            return False

    def __lt__(self,o):
        if self.id[0] > o.id[0]:
            return True
        elif self.id[0] < o.id[0]:
            return False
        else:
            return self.id[1] > o.id[1]

class Action(object): 
    def __init__(self,high_level_state1, high_level_state2,heuristic):
        self.src = high_level_state1
        self.dest = high_level_state2
        self.heuristic = heuristic
    
    def __str__(self):
        return("Action from {} to {}".format(self.source,self.dest))


class Sampler():
    def __init__(self,distribution,discretizer, mean = None, kd_tree = None):
        self.distribution = distribution
        self.discretizer = discretizer
        self.mean = mean
        self.kd_tree = kd_tree
    
    def sample(self):
        # xy_distribution = self.distribution[:,:,0].reshape(self.distribution.shape[0]*self.distribution.shape[1])
        xy_distribution = self.distribution.reshape(self.distribution.shape[0]*self.distribution.shape[1])
        xy_sample = np.random.choice(range(xy_distribution.shape[0]),p = xy_distribution)
        y_dof = int(xy_sample / self.distribution.shape[1])
        x_dof = xy_sample - int(y_dof * self.distribution.shape[1])
        start = 1 
        dof_samples = [x_dof,y_dof]        
        dof_values = self.discretizer.convert_sample(dof_samples)
        return dof_values
    
    def get_mean(self):
        if self.mean is not None:
            return self.mean
        else:
            raise NotImplementedError

    
    def get_distance_from_goal(self,ll_config):
        if self.kd_tree is None:
            raise NotImplementedError
        x = self.discretizer.get_bin_from_ll(ll_config[0],0)
        y = self.discretizer.get_bin_from_ll(ll_config[1],1)
        dist, index = self.kd_tree.query([y,x])
        return dist

class SingleValueSampler():

    def __init__(self,val):
        self.val = val
    
    def sample(self):
        return self.val

    def get_mean(self):
        return self.val

    def get_distance_from_goal(self,ll_config):
        return float((ll_config[0] - self.val[0]) ** 2 + (ll_config[1] - self.val[1]) ** 2) ** float(0.5)