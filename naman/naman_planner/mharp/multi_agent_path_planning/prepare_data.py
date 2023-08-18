import numpy as np 
import sys 
import os 
import yaml
import cv2
import tqdm
import pickle



class DataLoader(object): 
    def __init__(self, results_path, env_path):
        
        self.results_path = results_path
        self.env_path = env_path
        
        self.occupancy_matrix = self.occupancy_matrix_from_yaml()
        self.motion_plans = self.load_motion_plans()
    
    
    def occupancy_matrix_from_yaml(self):     
        with open(self.env_path,"r") as f: 
            map = yaml.load(f, Loader=yaml.FullLoader)
        obstacles = map["map"]["obstacles"]
        dimensions = np.asarray(map["map"]["dimensions"])
        occupancy_matrix = np.ones(shape = dimensions + 1)
        for i,j in obstacles: 
            occupancy_matrix[i,j] = 0
        return occupancy_matrix
    
    def load_motion_plans(self):
        if os.path.isfile("motion_plans.pickle"): 
            with open("motion_plans.pickle","rb") as f: 
                motion_plans = pickle.load(f)
        else:
            motion_plans = []
            l = os.listdir(self.results_path)
            print("Loading motion plans...")
            pbar = tqdm.tqdm(total = len(l))
            for result_file in l: 
                if ".yaml" in result_file: 
                    with open(os.path.join(self.results_path,result_file),"r") as f: 
                        data = yaml.load(f,Loader = yaml.FullLoader)
                    for agent in data["schedule"]: 
                        mp = [] 
                        for wp in data["schedule"][agent]: 
                            mp.append([wp["x"],wp["y"]])
                        motion_plans.append(mp)
                    pbar.update(1)
            with open("motion_plans.pickle","wb") as f: 
                pickle.dump(motion_plans, f)
        return motion_plans
    
    def create_critical_regions(self):
        freqs = np.zeros(shape = self.occupancy_matrix.shape )
        print("Creating critical regions")
        pbar = tqdm.tqdm(total = len(self.motion_plans))
        for mp in self.motion_plans:
            flags = np.zeros(shape = freqs.shape)
            for wp in mp: 
                if flags[wp[0],wp[1]] == 0: 
                    freqs[wp[0],wp[1]] += 1
                    flags[wp[0],wp[1]] = 1
            pbar.update(1)
        freqs = freqs / len(self.motion_plans)
        crs = np.zeros(freqs.shape)
        mask  = np.ma.masked_where(freqs > 0.2,crs)
        crs[mask.mask] = 1.0 
        crs_image = np.stack([self.occupancy_matrix, self.occupancy_matrix, self.occupancy_matrix],axis = -1)
        freqs_image = np.stack([freqs,freqs,freqs], axis = -1)
        for i in range(freqs.shape[0]):
            for j in range(freqs.shape[1]): 
                if freqs[i,j] > 0.15: 
                    crs_image[i,j,:] = [1,0,0]
        cv2.imwrite("temp.png",crs_image * 255.0 )
        
        
        
if __name__ == "__main__": 
    results_path = sys.argv[1]
    env_path = sys.argv[2]
    dl = DataLoader(results_path,env_path)
    dl.create_critical_regions()
    

