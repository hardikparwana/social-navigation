import os 
import numpy as np
import pickle

# from src.discretizer import Discretizer
# from src.abstraction import InterfaceAbstraction
# from src.data_structures import State, Action
# from src.search import Search
# from src.motion_planner import HARP
# from src.monitor import Monitor
# 
from discretizer import Discretizer
from abstraction import InterfaceAbstraction
from data_structures import State, Action
from search import Search
from motion_planner import HARP
from monitor import Monitor

class MHARP(object): 
    
    def __init__(self,env,robot,problem_config):
        self.env = env
        self.robot = robot 
        self.problem_config = problem_config
        self.preds_path = self.problem_config["nn_preds_path"]
        self.env_mask_path = self.problem_config["env_mask_path"]
        self.goal_tolerance = float(self.problem_config["goal_tolerance"])
         
        self.n_xy_bins = int(self.problem_config["n_xy_bins"])
        self.n_dof_bins = int(self.problem_config["n_dof_bins"])
        
        self.discretizer = Discretizer(robot, self.n_xy_bins, n_dof_bins = self.n_dof_bins)
        
        self.abstraction_path = self.problem_config["abstraction_path"]
        abs_fname = "{}_abs.p".format(self.problem_config['env_name'])
        if os.path.exists(os.path.join(self.abstraction_path,abs_fname)):
            with open(os.path.join(self.abstraction_path,abs_fname),"rb") as f:
                self.abstraction = pickle.load(f)
        else:
            self.nn_preds = self.load_pd()
            self.env_mask = self.load_env_mask()
            self.abstraction = InterfaceAbstraction(self.nn_preds,self.discretizer,self.env_mask,self.problem_config["criticality_threshold"])
            self.abstraction.kd_tree = None
            self.abstraction.centroids_kd_tree = None
            self.abstraction.interface_kd_tree = None
            self.abstraction.individual_kd_trees = None
            with open(os.path.join(self.abstraction_path,abs_fname),"wb") as f:
                pickle.dump(self.abstraction,f)
        self.abstraction.plot()
        self.actions = self.__initialize_actions()
        self.trainer = None
        self.monitor = Monitor(self.abstraction.abstract_states.values())

    def load_pd(self):
        pd = np.load(self.preds_path)
        pd = np.squeeze(pd)
        return pd
    
    def load_env_mask(self):
        env_mask = np.squeeze(np.load(self.env_mask_path))[:,:,0]
        return env_mask
                
    def __initialize_actions(self):
        actions = {}
        for src in self.abstraction.get_states():
            actions[src.id] = {}
            for dest_id in self.abstraction.reachability[src.id]:
                dest = self.abstraction.abstract_states[dest_id]
                actions[src.id][dest_id] = Action(src,dest,None)
        return actions
    
    def get_plan(self,init_config, goal_config, robot_id):
        init_xy = init_config[:-1]
        goal_xy = goal_config[:-1]
        init_state = self.abstraction.get_abstract_state(init_xy)
        goal_state = self.abstraction.get_abstract_state(goal_xy)
        print "computing high-level plan"
        high_level_plan = Search.gbfs(init_state, goal_state, self.actions, self.abstraction, self.monitor)
        self.abstraction.plot_plan(high_level_plan)
        sampling_region = self.abstraction.get_combined_hl_states_sampler(high_level_plan)
        critical_region_predictor = self.abstraction.get_combined_critical_region_sampler(high_level_plan)
        hl_plan_state_ids = [state.id for state in high_level_plan]
        print hl_plan_state_ids
        print "refining a high-level plan"
        motion_plan_found, motion_plan = HARP(self.env, init_config, goal_config, self.abstraction, self.discretizer, hl_plan_state_ids, sampling_region, critical_region_predictor, visualize=self.problem_config["visualize"]).get_mp()
        self.monitor.register(robot_id,high_level_plan)
        if motion_plan_found: 
            self.abstraction.plot_motion_plan(motion_plan)
            return motion_plan
        else: 
            return []