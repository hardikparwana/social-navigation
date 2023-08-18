import yaml
import sys
import os 

from src.robot import RobotConfig
from src.MHARP import MHARP
import openravepy as orpy
import numpy as np

def parse_yaml(yaml_file):
    with open(yaml_file,"r") as stream: 
        config = yaml.safe_load(stream)
        return config 
    
def prepare_robot(robot,llimits,ulimits):
    robot.SetActiveDOFs([0,1])
    robot.SetActiveDOFVelocities([0.5,0.5,0.5])
    robot.SetDOFLimits(llimits,ulimits)
    
def prepare_env(problem_config,start,goal):
    env = orpy.Environment()
    env.Load("./3denvs/{}.stl".format(problem_config["env_name"]))
    room = env.GetKinBody("env1")
    t = room.GetTransform()
    t[2,3] += 1
    room.SetTransform(t)
    or_robot = env.ReadRobotXMLFile("./3denvs/{}".format(problem_config["robot"]["model_path"]))
    or_robot.SetName(problem_config["robot"]["name"])
    env.Add(or_robot)
    t = np.eye(4)
    t[2,3] += 1.05
    or_robot.SetTransform(t)
    prepare_robot(or_robot,problem_config["robot"]["llimits"],problem_config["robot"]["ulimits"])

    or_robot.SetActiveDOFValues(start)
    cc = orpy.RaveCreateCollisionChecker(env,"pqp")
    return env
    
        
if __name__ == "__main__": 
    
    problem_config = parse_yaml(sys.argv[1])
    
    robot = RobotConfig(problem_config["robot"]["name"], problem_config["robot"]["ndofs"], problem_config["robot"]["llimits"], problem_config["robot"]["ulimits"])
    # robot = Robot("a",3,[1,2,3], [4,5,6])
    
    if problem_config["reset"]: 
        os.system("rm -rf abstraction/*.p")

    start1 = [-8,-8]
    goal1 = [8,6]
    
    start2 = [-8,-8]
    goal2 = [8,6]
    
    start3 = [8,6]
    goal3 = [-8,-8]
    
    env = prepare_env(problem_config,start1,goal1)
    if problem_config["visualize"]:
        env.SetViewer('qtcoin')
    # env = None

    mharp = MHARP(env,robot,problem_config)
    mp_plan = mharp.get_plan(start1,goal1,"robot1")
    mharp.monitor.go_to_next_state("robot1")
    mp_plan = mharp.get_plan(start2,goal2,"robot2")
    mharp.monitor.go_to_next_state("robot2")
    mp_plan = mharp.get_plan(start2,goal2,"robot3")
    # mp_plan = mharp.get_plan(start3,goal3,"robot3")
    print(mp_plan) #! TODO: execute this plan
    
    