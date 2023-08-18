
import copy
import math
from random import random
from re import L
from scipy.ndimage.measurements import label
import collections
import numpy as np
from scipy.spatial import KDTree
import cv2
import tqdm
from collections import Counter


from discretizer import Discretizer
from data_structures import Interface, State, Sampler



class Abstraction(object): 
    
    def __init__(self,nn_preds, discretizer, env_mask, criticality_threshold = 0.6 ):
        self.nn_preds = nn_preds
        self.discretizer = discretizer
        self.env_mask = env_mask
        self.criticality_threshold = criticality_threshold
        self.xy_preds = self.nn_preds[:,:,0]
        self.centroids = {}
        self.rbvd_image = None
        
        self.criticality_mask = self.xy_preds > self.criticality_threshold
        self.abstract_preds = copy.deepcopy(self.xy_preds)
        self.abstract_preds[self.criticality_mask] = 1.0
        self.abstract_preds[~self.criticality_mask] = 0.0
        self.abstract_preds *= self.env_mask
        self.abstraction, self.number_of_abstract_states = label(self.abstract_preds)
        self.abstract_state_ids = list(range(1,self.number_of_abstract_states+1))
        self.__process_abstractions() 
        self.number_of_abstract_states = len(self.abstract_state_ids)
        self.kd_tree, self.centroids_kd_tree, self.individual_kd_trees = self.__create_kd_tree()
        self.abstract_states = self.__make_abstract_states()
        self.complete_rbvd = self.__make_complete_rbvd()
        self.old_rbvd = copy.deepcopy(self.complete_rbvd)
        self.complete_rbvd, self.kd_tree, self.centroids_kd_tree, self.number_of_abstract_states = self.__process_rbvds()
        assert self.number_of_abstract_states == len(self.abstract_state_ids)
        self.reachability = self.__compute_reachability()
        self.image = None 
        
    def get_combined_critical_region_sampler(self,abstract_states): 
        state_ids = [state.id for state in abstract_states]
        abstraction = copy.deepcopy(self.abstraction)
        indices = np.in1d(abstraction,state_ids).reshape(abstraction.shape)
        abstraction[indices] = 1.0
        abstraction[~indices] = 0.0
        abstraction = abstraction / float(np.sum(abstraction))
        sampler = Sampler(abstraction, self.discretizer)
        return sampler
    
    def get_combined_hl_states_sampler(self,abstract_states):
        state_ids = [state.id for state in abstract_states]
        abstraction = copy.deepcopy(self.complete_rbvd)
        indices = np.in1d(abstraction,state_ids).reshape(abstraction.shape)
        abstraction[indices] = 1.0
        abstraction[~indices] = 0.0
        abstraction = abstraction / float(np.sum(abstraction))
        sampler = Sampler(abstraction, self.discretizer)
        return sampler
        
    
    def __compute_reachability(self):
        reachability = {}
        added_transitions = set()
        pbar = tqdm.tqdm(total = (self.complete_rbvd.shape[0] - 1) * (self.complete_rbvd.shape[1] -1 )) 
        for i in range(self.complete_rbvd.shape[0]-1):
            for j in range(self.complete_rbvd.shape[1] - 1):
                if (self.complete_rbvd[i,j] != self.complete_rbvd[i,j+1] and self.complete_rbvd[i,j] != 0 and self.complete_rbvd[i,j+1] != 0): 
                    if ((self.complete_rbvd[i,j],self.complete_rbvd[i,j+1]) not in added_transitions and (self.complete_rbvd[i,j+1],self.complete_rbvd[i,j]) not in added_transitions) :
                        if self.complete_rbvd[i,j] not in reachability:
                            reachability[self.complete_rbvd[i,j]] = set()
                        if self.complete_rbvd[i,j+1] not in reachability:
                            reachability[self.complete_rbvd[i,j+1]] = set()
                        reachability[self.complete_rbvd[i,j]].add(self.complete_rbvd[i,j+1])
                        reachability[self.complete_rbvd[i,j+1]].add(self.complete_rbvd[i,j])
                if (self.complete_rbvd[i,j] != self.complete_rbvd[i+1,j] and self.complete_rbvd[i,j] != 0 and self.complete_rbvd[i+1,j] != 0): 
                    if ((self.complete_rbvd[i,j],self.complete_rbvd[i+1,j]) not in added_transitions and (self.complete_rbvd[i+1,j],self.complete_rbvd[i,j]) not in added_transitions) :
                        if self.complete_rbvd[i,j] not in reachability:
                            reachability[self.complete_rbvd[i,j]] = set()
                        if self.complete_rbvd[i+1,j] not in reachability:
                            reachability[self.complete_rbvd[i+1,j]] = set()
                        reachability[self.complete_rbvd[i,j]].add(self.complete_rbvd[i+1,j])
                        reachability[self.complete_rbvd[i+1,j]].add(self.complete_rbvd[i,j])
                pbar.update(1)
        return reachability 
    
    def __make_complete_rbvd(self):
        pbar = tqdm.tqdm(total = self.abstraction.shape[0]*self.abstraction.shape[1])
        complete_rbvd = np.zeros(shape = self.abstraction.shape)
        for i in range(self.abstraction.shape[0]):
            for j in range(self.abstraction.shape[1]):
                if self.env_mask[i,j] == 1:
                    dist,index = self.kd_tree.query([i,j])
                    idx, idy = self.kd_tree.data[index]
                    complete_rbvd[i,j] = self.abstraction[int(idx),int(idy)]
                pbar.update(1)
        return complete_rbvd
    
    def __process_rbvds(self):
        new_num_abs_state = max(self.abstract_state_ids)
        new_rbvd = copy.deepcopy(self.complete_rbvd)
        centroid_kd_tree_data = list(self.centroids_kd_tree.data)
        kd_tree_data = list(self.kd_tree.data)
        pbar = tqdm.tqdm(total = len(self.abstract_state_ids))
        new_state_ids = copy.deepcopy(self.abstract_state_ids)
        for i in self.abstract_state_ids:
            rbvd_copy = copy.deepcopy(self.complete_rbvd)
            mask = np.ma.masked_where(rbvd_copy == i, rbvd_copy)
            rbvd_copy[~mask.mask] = 0.0
            rbvd_copy[mask.mask] = 1.0
            hl_labels, num = label(rbvd_copy)
            if num > 1: 
                '''
                get the actual high-level state where it belongs
                '''
                ids = np.argwhere(self.abstraction == i)
                set_ids =set(tuple([tuple(id) for id in ids ]))
                original_cr = None
                for n in range(1,num+1):
                    selected_ids =set(tuple([tuple(id) for id in np.argwhere(hl_labels == n) ]))
                    if len(set_ids.intersection(selected_ids)) > 0:
                        original_cr = n
                        break
                    # mean_temp = np.mean(selected_ids,axis=0).astype(np.int)
                    # idx_temp, idy_temp = mean_temp[0], mean_temp[1]
                    # if (ids_x - idx_temp)**2 + (ids_y - idy_temp)** 2 < min_dist:
                    #     original_cr = n 
                    #     min_dist = (ids_x - idx_temp)**2 + (ids_y - idy_temp)** 2 
                '''
                keep the original state and CR as it is and create new states for other CRs. 
                '''

                for n in range(1,num+1):
                    if n == original_cr:
                        continue
                    else:
                        mask = np.ma.masked_where(hl_labels == n, new_rbvd)
                        
                        '''
                        sample few points for cr for new generated hl_state
                        '''
                        idx = np.argwhere(mask.mask)
                        
                        if 0.15 *len(idx) > 35: # To change:
                        # if len(idx) > 35: # To change:
                            new_id = new_num_abs_state + 1
                            new_rbvd[mask.mask] = int(new_id)
                            new_num_abs_state+= 1
                            sidx =  np.random.choice(range(idx.shape[0]),int(idx.shape[0] * 0.15))
                            sids = idx[sidx]
                            for x,y in sids:
                                self.abstraction[x,y] = new_id
                                # self.abstraction[y,x] = new_id
                                kd_tree_data.append([x,y])
                                # kd_tree_data.append([y,x])
                            self.individual_kd_trees[new_id] = KDTree(sids)
                            # new_centroids = np.median(sids,axis = 0).astype(np.int)
                            new_centroids = self.median(sids).astype(np.int32)
                            new_cents = [new_centroids[1],new_centroids[0]]
                            self.centroids[new_id] = new_cents
                            centroid_kd_tree_data.append(new_cents)
                            self.abstract_states[new_id] = State(new_id,new_cents)
                            new_state_ids.append(new_id)
                        else:
                            new_rbvd[mask.mask] = 0
            pbar.update(1)

        new_kd_tree = KDTree(kd_tree_data)
        new_centroids_kd_tree = KDTree(centroid_kd_tree_data)
        self.abstract_state_ids = new_state_ids
        return new_rbvd, new_kd_tree, new_centroids_kd_tree, len(self.abstract_state_ids)
    
    def __process_abstractions(self):
        counts = collections.Counter(list(self.abstraction.flatten()))
        for n in range(self.number_of_abstract_states+1):
            if counts[n] < 150:  # TODO: some better approach to do this.
                idx = np.where(self.abstraction == n)
                self.abstraction[idx] = 0
                self.abstract_state_ids.remove(n)
    
    def __make_abstract_states(self):
        abstract_states = {}
        for abstract_state_id in self.abstract_state_ids:
            # idx = np.argwhere(self.abstraction == abstract_state_id)
            s = State(abstract_state_id,self.centroids[abstract_state_id])
            abstract_states[abstract_state_id] = s
        
        return abstract_states

    def sfun(self,a):
        return a[0] + a[1]

    def median(self,arr):
        if type(arr) is not type([]):
            alist = arr.tolist()
        else:
            alist = arr
        # sorted_arr = np.array(sorted(alist))
        sorted_arr = np.array(sorted(alist,key =  self.sfun))
        return sorted_arr[int(sorted_arr.shape[0]/2)].astype(np.int32)
    
    def __create_kd_tree(self):
        idx = np.argwhere(self.abstraction)
        kd_tree = KDTree(idx)
        individual_kd_trees = {}
        for abstract_state_id in self.abstract_state_ids:
            idx = np.argwhere(self.abstraction == abstract_state_id)
            # mean = np.median(idx,axis = 0).astype(np.int)
            mean = self.median(idx).astype(np.int32)  
            self.centroids[abstract_state_id] = [mean[1],mean[0]]
            individual_kd_trees[abstract_state_id] = KDTree(idx)
        centroids_kd_tree = KDTree(list(self.centroids.values()))
        return kd_tree, centroids_kd_tree, individual_kd_trees
    
    def get_abstract_state(self,ll_config):
        x = self.discretizer.get_bin_from_ll(ll_config[0],0)
        y = self.discretizer.get_bin_from_ll(ll_config[1],1)
        state_id = self.complete_rbvd[y,x]
        if state_id == 0.0 or state_id == 0: 
            return State(0,None)
        return self.abstract_states[state_id]
    
    def get_abstract_state_volume(self,abstract_state): 
        idx = self.complete_rbvd == abstract_state.id 
        return np.sum(idx)
        
    def check_env_collision(self,ll_config):
        x = self.discretizer.get_bin_from_ll(ll_config[0],0)
        y = self.discretizer.get_bin_from_ll(ll_config[1],1)
        return self.env_mask[y,x] == 0

    def euclidean_distance(self,c1,c2):
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        return np.linalg.norm(c1-c2)

    def estimate_heuristic(self,s1,s2):
        c1 = s1.get_centroid()
        c2 = s2.get_centroid() 
        p1 = self.discretizer.convert_sample(c1)
        p2 = self.discretizer.convert_sample(c2)
        return self.euclidean_distance(p1,p2)
    
    def get_uniform_sampler(self,discretizer = None):
        distribution = copy.deepcopy(self.xy_preds)
        abstraction = np.ones(self.abstraction.shape)
        abstraction = abstraction / float(np.sum(abstraction))
        distribution = abstraction
        if discretizer is None:
            sampler = Sampler(distribution,self.discretizer)
        else:
            sampler = Sampler(distribution,discretizer)
        return sampler

    def get_eval_sampling_distribution(self):
        distribution = np.ones(self.network_preds.shape)
        distribution = distribution / np.sum(distribution)
        sampler = Sampler(distribution,self.discretizer)
        return sampler

    def plot(self):
        env_image = np.expand_dims(copy.deepcopy(self.env_mask),axis = 2)
        env_image = np.concatenate([env_image,env_image,env_image],axis = -1)
        colors = {}
        for state in self.abstract_states.values():
            mask = self.abstraction == state.id 
            color = np.asarray([np.random.random(),np.random.random(),np.random.random()])
            env_image[mask] = color
            colors[state.id] = color
            x,y = state.get_centroid()[:2]
            cv2.putText(env_image,str(state.id),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 2, cv2.LINE_AA)
        image = env_image* 255.0
        cv2.imwrite("temp.png",image)
        return colors
        
class InterfaceAbstraction(Abstraction):

    def __init__(self,network_prediction, discretizer, env_mask, criticality_threshold = 0.6):

        super(InterfaceAbstraction,self).__init__(network_prediction, discretizer, env_mask, criticality_threshold)
        self.interfaces, self.interface_kd_tree = self.__generate_interfaces()
        self.interface_regions = self.__create_interface_regions()
        self.interface_connectivity = self.__compute_interface_connectivity()
        self.connectivity = self.interface_connectivity
        print("Interface abstraction generated.")

    def __compute_interface_connectivity(self):
        interface_connectivity = {}
        for interface in self.interfaces:
            temp = set([ frozenset(intf) for intf in self.interfaces if len(set(interface).intersection(set(intf))) > 0 and intf != interface and intf != interface[::-1] ])
            interface_connectivity[interface] = [tuple(intf) for intf in temp]
        return interface_connectivity

    def __create_interface_regions(self):
        interface_regions = np.expand_dims(np.zeros(self.abstraction.shape),axis = -1)
        interface_regions = np.stack([interface_regions,interface_regions],axis = -1)
        visited = set()
        for interface in self.interfaces:
            if interface not in visited and (interface[1],interface[0]) not in visited:
                centroid = self.interfaces[interface].get_centroid()
                width = math.ceil(0.2 * self.discretizer.n_xy_bins / 75.0) #TODO: for 41,42,43 this has to be set to 25.0 

                for x in range(int(max(centroid[0]-width,0)), int(min(centroid[0]+width + 1,self.discretizer.n_xy_bins))):
                    for y in range(int(max(centroid[1]-width,0)), int(min(centroid[1]+width+1,self.discretizer.n_xy_bins))):
                        # interface_regions[x,y] = interface
                        interface_regions[y,x] = interface
        return interface_regions

    def __generate_interfaces(self):
        reachability = {}
        added_transitions = set()
        interfaces = {}
        counts = collections.Counter()
        pbar = tqdm.tqdm(total = (self.complete_rbvd.shape[0] - 1) * (self.complete_rbvd.shape[1] -1 )) 
        for i in range(self.complete_rbvd.shape[0]-1):
            for j in range(self.complete_rbvd.shape[1] - 1):
                if (self.complete_rbvd[i,j] != self.complete_rbvd[i,j+1] and self.complete_rbvd[i,j] != 0 and self.complete_rbvd[i,j+1] != 0): 
                    if (self.complete_rbvd[i,j],self.complete_rbvd[i,j+1]) not in interfaces.keys():
                        interfaces[(self.complete_rbvd[i,j],self.complete_rbvd[i,j+1])] = []
                    if (self.complete_rbvd[i,j+1],self.complete_rbvd[i,j]) not in interfaces.keys():
                        interfaces[(self.complete_rbvd[i,j+1],self.complete_rbvd[i,j])] = []

                    interfaces[(self.complete_rbvd[i,j],self.complete_rbvd[i,j+1])].extend([(i,j),(i,j+1)])
                    interfaces[(self.complete_rbvd[i,j+1],self.complete_rbvd[i,j])].extend([(i,j),(i,j+1)])

                if (self.complete_rbvd[i,j] != self.complete_rbvd[i+1,j] and self.complete_rbvd[i,j] != 0 and self.complete_rbvd[i+1,j] != 0): 
                    if (self.complete_rbvd[i,j],self.complete_rbvd[i+1,j]) not in interfaces.keys():
                        interfaces[(self.complete_rbvd[i,j],self.complete_rbvd[i+1,j])] = []
                    if (self.complete_rbvd[i+1,j],self.complete_rbvd[i,j]) not in interfaces.keys():
                        interfaces[(self.complete_rbvd[i+1,j],self.complete_rbvd[i,j])] = []

                    interfaces[(self.complete_rbvd[i,j],self.complete_rbvd[i+1,j])].extend([(i,j),(i+1,j)])
                    interfaces[(self.complete_rbvd[i+1,j],self.complete_rbvd[i,j])].extend([(i,j),(i+1,j)])
                    
                pbar.update(1)
        
        interfaces_objs, interface_kd_tree = self.__make_interfaces(interfaces)
        return interfaces_objs, interface_kd_tree

    def get_other_DOF_means(self,pair):
        centroids1 = self.centroids[pair[0]]
        centroids2 = self.centroids[pair[1]]
        new_centroids = [] 
        for i in range(2,len(centroids1)):
            new_centroids.append(int((centroids1[i]+centroids2[i])/2))
        return new_centroids

    def __make_interfaces(self,interfaces):
        medians = {}
        interface_objs = {}
        interface_kd_trees_data = []
        for pair in interfaces.keys():
            if len(interfaces[pair]) > 18:
                medians[pair] = self.median(interfaces[pair])
                centroids = [medians[pair][1],medians[pair][0]]
                other_dof_means = self.get_other_DOF_means(pair)
                centroids.extend(other_dof_means)
                # interface_objs[pair] = Interface(pair,[medians[pair][1],medians[pair][0]])
                interface_objs[pair] = Interface(pair,centroids)
                # interface_kd_trees_data.append(medians[pair]) check the effect or otherwise rollback
                interface_kd_trees_data.append(centroids)
        interface_kd_tree = KDTree(interface_kd_trees_data)
        return interface_objs, interface_kd_tree
    
    def get_closest_region(self,ll_config):
        binned_point = []
        for i in range(len(ll_config)):
            binned_point.append(self.discretizer.get_bin_from_ll(ll_config[i],i))
        # binned_point[0],binned_point[1]  = binned_point[1], binned_point[0]
        dist, index = self.interface_kd_tree.query(binned_point,5)
        hl_state_id = self.get_abstract_state(ll_config).id
        for i in index:
            c_x,c_y = self.interface_kd_tree.data[i][:2]
            interface_str = self.interface_regions[int(c_y),int(c_x)]
            interface = self.interfaces[tuple(interface_str.tolist()[0])]
            if hl_state_id in interface.id:
                return interface

    def get_state_from_id(self, state_id):
        return self.interfaces[state_id]

    def get_states(self):
        return [state for state in self.abstract_states.values()]

    def plot(self):
        colors = super(InterfaceAbstraction,self).plot()
        env_image = np.expand_dims(copy.deepcopy(self.env_mask),axis = 2)
        env_image = np.concatenate([env_image,env_image,env_image],axis = -1)
        for state in self.abstract_states.values():
            mask = self.complete_rbvd == state.id
            env_image[mask] = colors[state.id]
        drawn = set()
        for i1 in self.interfaces:
            if (i1[0],i1[1]) not in drawn and (i1[1],i1[0]) not in drawn:
                x,y = self.interfaces[i1].get_centroid()[:2]
                cv2.circle(env_image,(x,y),5,(1,1,1),1)


        self.rbvd_image = env_image * 255.0
        cv2.imwrite("rbvd.png",self.rbvd_image)
        
        
    def plot_plan(self,abstract_states):
        if self.rbvd_image is None: 
            self.plot() 
        env_image = self.rbvd_image
        for i in range(0,len(abstract_states)-1):
            state1 = abstract_states[i]
            state2 = abstract_states[i+1]
            x1,y1 = state1.get_centroid()[:2]
            x2,y2 = state2.get_centroid()[:2]
            cv2.putText(env_image,str(state1.id),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 2, cv2.LINE_AA)
            cv2.line(env_image,(x1,y1),(x2,y2),[0,0,0],1,cv2.LINE_AA)
        cv2.putText(env_image,str(state2.id),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 2, cv2.LINE_AA)
        cv2.imwrite("plan.png",env_image)      
    
    def plot_motion_plan(self,motion_plan):
        wps = []
        wps_set = set()
        img = self.rbvd_image
        for wp in motion_plan:
            x = self.discretizer.get_bin_from_ll(wp[0],0)
            y = self.discretizer.get_bin_from_ll(wp[1],1)
            if (x,y) not in wps_set:
                wps_set.add((x,y))
                wps.append((x,y))
        
        for i in range(len(wps)-1):
            x1,y1 = wps[i]
            x2,y2 = wps[i+1]
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)), [255,255,255], 1, cv2.LINE_AA)
            
        cv2.imwrite("mp.png",img)
                  

    def get_critical_region_sampler(self,option,discretizer = None):
        abstraction = copy.deepcopy(self.abstraction)
        region_id = list(set(option.src.id).intersection(set(option.dest.id)))[0]
        mask = abstraction == region_id
        new_abstraction = np.zeros(abstraction.shape)
        new_abstraction[mask] = 1.0
        abstraction = copy.deepcopy(new_abstraction)
        abstraction = abstraction / float(np.sum(abstraction))
        distribution = abstraction
        if discretizer is None:
            sampler = Sampler(distribution,self.discretizer)
        else:
            sampler = Sampler(distribution,discretizer)
        return sampler
    
    def get_abstract_state_sampler(self,option, discretizer = None):
        rbvd = copy.deepcopy(self.complete_rbvd)
        new_rbvd = np.zeros(rbvd.shape)
        if type(option.src.id) == type(tuple()) and type(option.dest.id) == type(tuple()):
            state_id = list(set(option.src.id).intersection(set(option.dest.id)))[0]
        elif type(option.src.id) != type(tuple()):
            state_id = option.src.id
        elif type(option.dest.id) != type(tuple()):
            state_id = option.dest.id
        mask1 = rbvd == state_id
        new_rbvd[mask1] = 1.0
        rbvd = new_rbvd / float(np.sum(new_rbvd))
        distribution = rbvd
        if discretizer is None:
            sampler = Sampler(distribution,self.discretizer)
        else:
            sampler = Sampler(distribution,discretizer)
        return sampler

    def get_sampler(self,interface, discretizer=None, mode = 1):
        centroid = interface.get_centroid()
        width = math.ceil(0.2 * self.discretizer.n_xy_bins / 75.0) #TODO: for 41,42,43 this has to be set to 25.0 
        abstraction = np.zeros(shape = self.network_preds.shape)
        for x in range(max(centroid[0]-width,0),min(centroid[0]+width+1,self.discretizer.n_xy_bins)):
            for y in range(max(centroid[1]- width,0),min(centroid[1]+width+1,self.discretizer.n_xy_bins)):
                if self.complete_rbvd[y,x] in interface.id:
                    abstraction[y,x] = 1.0
        abstraction = abstraction / float(np.sum(abstraction[:,:,0]))

        # distribution = copy.deepcopy(self.network_preds)
        # distribution[:,:,0] = abstraction
        distribution = abstraction.copy()
        if discretizer is None:
            sampler = Sampler(distribution,self.discretizer,mean = interface.get_centroid())
        else:
            sampler = Sampler(distribution,discretizer,mean = interface.get_centroid())
            
        return sampler

    def in_region(self,ll_config,interface_region):
        x = self.discretizer.get_bin_from_ll(ll_config[0],0)
        y = self.discretizer.get_bin_from_ll(ll_config[1],1)
        other_dofs = []
        for i in range(2,len(ll_config)):
            other_dofs.append(self.discretizer.get_bin_from_ll(ll_config[i],i))
        if set(self.interface_regions[y,x].tolist()[0]) == set(interface_region.id):
            return True
        else:
            return False

    def set_evaluation_function(self,s1,s2):
        if isinstance(s1, Interface) and isinstance(s2, Interface):
            self.eval = {'type':1, 'src':s1, 'dest':s2}
        elif isinstance(s2, Interface):
            s1 = self.get_abstract_state(s1)
            self.eval = {'type':2, 'src':s1, 'dest':s2}
        else:
            self.eval = {'type':3, 'src':s1, 'dest':s2}


    def conv_check(self,ll_config,target):
        ll_config = np.asarray(ll_config)
        target = np.asarray(target)
        lin_distance = np.linalg.norm(ll_config[:2] - target[:2])
        # dof_vector = np.abs(ll_config[2:] - target[2:]) 
        if lin_distance < 0.4:  # and (dof_vector < 0.5).all():          
            return True
        else:
            return False 

    def evaluate_function(self, llconfig):
        '''
        Types:
        type 1: Both source and destination are regions
        type 2: Destination is a region, source is a low level raw configuration
        type 3: Both source and destination are low level raw configurations
        '''
        if self.eval['type'] == 1:
            try:
                abstract_state = self.get_abstract_state(llconfig)
            except: # WTF hack to avoid rbvd zeroed out state at the wall edges
                return -2 
            # if self.conv_check(llconfig,self.discretizer.convert_sample(self.eval['dest'].get_centroid())) and abstract_state.id in self.eval["dest"].id:
            if self.in_region(llconfig,self.eval["dest"]):
                return 1
            # if abstract_state.id == self.eval["src_set"].intersection(set(self.eval["dest"].id)):
            # if abstract_state.id in self.eval["dest"].id or abstract_state.id in self.eval["src"].id:
            if abstract_state.id in self.eval["dest"].id or self.in_region(llconfig,self.eval["src"]):
                return 0
            else:
                return -1

        elif self.eval['type'] == 2:
            try:
                abstract_state = self.get_abstract_state(llconfig)
            except: # WTF hack to avoid rbvd zeroed out state at the wall edges
                return -2 
            # if self.conv_check(llconfig,self.discretizer.convert_sample(self.eval['dest'].get_centroid())) and abstract_state.id in self.eval["dest"].id:
            if self.in_region(llconfig,self.eval["dest"]):
                return 1
            # if abstract_state.id == self.eval["src_set"].intersection(set(self.eval["dest"].id)):
            if abstract_state.id in self.eval["dest"].id or abstract_state.id == self.eval["src"].id:
                return 0
            else:
                return -1
        else:
            try:
                abstract_state = self.get_abstract_state(llconfig)
                goal_abstract_state = self.get_abstract_state(self.eval["dest"])
            except: # WTF hack to avoid rbvd zeroed out state at the wall edges
                return -2 
            if self.eval['type'] == 3:
                if self.conv_check(llconfig,self.eval['dest']) and abstract_state == goal_abstract_state: #TODO: change it to parameter
                    return 1
                elif abstract_state == goal_abstract_state or self.in_region(llconfig,self.eval["src"]):
                    return 0
                else:
                    return -1

    __call__ = evaluate_function
