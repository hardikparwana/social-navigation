import numpy as np

class RobotConfig():

    def __init__(self,name,ndofs,llimits,ulimits):
        self.name = name
        self.ndofs = ndofs
        self.llimits = np.array(llimits)
        self.ulimits = np.array(ulimits)
    
    def get_dof_lower_limits(self):
        return self.llimits
    
    def get_dof_upper_limits(self):
        return self.ulimits

    def get_num_active_dofs(self):
        return self.ndofs