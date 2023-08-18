import yaml
import sys 
import numpy as np 

from itertools import product

template_yaml_path = sys.argv[1]
yaml_folder = sys.argv[2]
n_agents = int(sys.argv[3])
n_problems = int(sys.argv[4])
# 
# template_yaml_path = "../benchmark/custom/template.yaml"
# yaml_folder = "../benchmark/custom/"
# n_agents = 1
# n_problems = 1


def manhatten_distance(a,b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


with open(template_yaml_path,"r") as f: 
    template_yaml = yaml.load(f, Loader=yaml.FullLoader)
    


obstacles = template_yaml["map"]["obstacles"]
all_pairs = list(product(list(range(template_yaml["map"]["dimensions"][0])),list(range(template_yaml["map"]["dimensions"][1]))))


for p_num in range(n_problems):
    starts = []
    agents = []
    goals = []
    for a_num in range(n_agents):
        while True: 
            while True: 
                start = all_pairs[np.random.choice(list(range(len(all_pairs))))]
                if start not in obstacles and start not in starts:
                    break
            while True: 
                goal = all_pairs[np.random.choice(list(range(len(all_pairs))))]
                if goal not in obstacles and goal not in goals:
                    break
            
            if manhatten_distance(start,goal) > 15:
                starts.append(start)
                goals.append(goal)
                break        
            
        name = "agent"+str(a_num)
        agents.append({"start":start, "goal":goal, "name": name})
    template_yaml["agents"] = agents
    template_yaml["dynamic_obstacles"] = {}
    with open(yaml_folder+"problem_{}.yaml".format(p_num),"w") as f:
        yaml.dump(template_yaml,f)