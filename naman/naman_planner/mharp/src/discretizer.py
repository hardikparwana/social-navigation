class Discretizer(object):

    def __init__(self,robot,n_xy_bins = 224,n_dof_bins = 10):
        self.robot = robot
        self.n_dofs = self.robot.get_num_active_dofs()
        self.n_xy_bins = n_xy_bins
        self.n_dof_bins = n_dof_bins
        self.bins = self.__create_bins()

    def __create_bins(self):
        bins = {}
        llimits = self.robot.get_dof_lower_limits()
        ulimits = self.robot.get_dof_upper_limits()
        dof_range = ulimits - llimits
        
        # for x-dof
        i = 0
        bins[i] = {}
        start = llimits[i]
        bin_step = dof_range[i] / float(self.n_xy_bins)
        bins[i]["bin_start"] = []
        bins[i]["bin_end"] = []
        for j in range(self.n_xy_bins):
            bins[i]["bin_start"].append(start)
            bins[i]["bin_end"].append(start + bin_step)
            start += bin_step


        # for y-dof
        i = 1
        bins[i] = {}
        bins[i]["bin_start"] = []
        bins[i]["bin_end"] = []
        start = ulimits[i]
        bin_step = dof_range[i] / float(self.n_xy_bins)
        for j in range(self.n_xy_bins):
            bins[i]["bin_start"].append(start - bin_step)
            bins[i]["bin_end"].append(start)
            start -= bin_step
        return bins

    def get_bins(self):
        return self.bins
    
    def convert_sample(self,dof_sample):
        dof_values = []
        for i in range(len(dof_sample)):
            dof_value = (self.bins[i]['bin_start'][dof_sample[i]] + self.bins[i]['bin_end'][dof_sample[i]]) / 2.
            dof_values.append(dof_value)
        return dof_values

    def get_bin_from_ll(self, dofval, jointIdx):
        
        bins = self.bins[jointIdx]
        if jointIdx == 1:
            if dofval > bins['bin_start'][0]:
                return 0
            if dofval < bins['bin_end'][-1]:
                return len(bins['bin_end'])-1
        else:
            if dofval < bins['bin_start'][0]:
                return 0
            if dofval > bins['bin_end'][-1]:
                return len(bins['bin_end'])-1
        for j in range(len(bins['bin_start'])):
            if bins['bin_start'][j] <= dofval <= bins['bin_end'][j]:
                return j
