#! /usr/bin/python 

import rospy 
from MHARP import MHARP
import openravepy as orpy
import yaml
from robot import RobotConfig
import os
from mharp_msgs.srv import GetMHARPMotionPlan, GetMHARPMotionPlanResponse
from mharp_msgs.srv import StartServer, StartServerResponse
from mharp_msgs.msg import WayPoint
import numpy as np
from gazebo_msgs.msg import ModelStates

import os 
# os.chdir("/root/catkin_ws/src/mharp/")
os.chdir("/home/naman/catkin_ws/src/naman_planner/mharp/")

class ROSService(object): 

    def __init__(self,mharp,n_robots):
        rospy.init_node("ros_mharp_node")
        self.mharp = mharp 
        self.n_robots = n_robots
        rospy.Service("get_mharm_motion_plan", GetMHARPMotionPlan, self.get_motion_plan)
        rospy.Service("start_motion_planning_server", StartServer,self.create_instance)
        self.robot_model_subscriber = rospy.Subscriber("/gazebo/get_model_states/",ModelStates,callback=self.robot_model_pose_callback)
        self.robot_poses = {}
        self.robot_abstract_states = {} 
        rospy.spin()
    
    
    def robot_model_pose_callback(self,data): 
        for i in range(len(data.name)): 
            if data.name[i] in self.robot_poses: 
                x = data.pose.position.x
                y = data.pose.position.y
                self.robot_poses[data.name[i]] = [x,y]
                   
    def get_motion_plan(self,req): 
        robot_id = req.robot_id
        current_config = req.start
        goal_config = req.goal
        robot = self.mharp.env.GetRobots()[0]
        robot.SetActiveDOFValues(current_config)
        mp_plan = self.mharp.get_plan(current_config,goal_config,robot_id)
        mp_msg = []
        for wp in mp_plan:
            mp_msg.append(WayPoint(wp))
        return GetMHARPMotionPlanResponse(mp_msg)
        
    def monitor_robots(self):
        for robot in self.robot_abstract_states: 
            current_abs_state = self.mharp.abstraction.get_abstract_state(self.robot_poses[robot]).id
            if self.robot_abstract_states[robot] is None: 
                self.robot_abstract_states[robot] = current_abs_state
            elif self.robot_abstract_states[robot] != current_abs_state: 
                self.mharp.monitor.go_to_next_state(robot)
                self.robot_abstract_states[robot] = current_abs_state
               
    def prepare_robot(self,robot,llimits,ulimits):
        robot.SetActiveDOFs([0,1])
        robot.SetActiveDOFVelocities([0.5,0.5,0.5])
        robot.SetDOFLimits(llimits,ulimits)
    
    def prepare_env(self,problem_config,start,goal):
        env = orpy.Environment()
        env.Load("./3denvs/{}.stl".format(problem_config["env_name"]))
        room = env.GetKinBody("env1")
        t = room.GetTransform()
        t[2,3] += 1
        room.SetTransform(t)
        robot_file = "./3denvs/{}".format(problem_config["robot"]["model_path"])
        if ".xml" in robot_file:
            or_robot = env.ReadRobotXMLFile("./3denvs/{}".format(problem_config["robot"]["model_path"]))
        elif ".urdf" in robot_file: 
            module = orpy.RaveCreateModule(env,"urdf")
            name = module.SendCommand("loadURI {} {}".format(robot_file, robot_file[:-4] + "srdf"))
            or_robot = env.GetRobot(name)
        or_robot.SetName(problem_config["robot"]["name"])
        env.Add(or_robot)
        t = np.eye(4)
        t[2,3] += 1.05 
        or_robot.SetTransform(t)
        self.prepare_robot(or_robot,problem_config["robot"]["llimits"],problem_config["robot"]["ulimits"])
        or_robot.SetActiveDOFValues(start)
        cc = orpy.RaveCreateCollisionChecker(env,"pqp")
        return env

    def parse_yaml(self,yaml_file):
        with open(yaml_file,"r") as stream: 
            config = yaml.safe_load(stream)
        return config 
    
    def create_instance(self,req):
        print "inside create instance"
        yaml_name = req.name
        problem_config = self.parse_yaml(yaml_name)
        start1 = [-8,-8]
        goal1 = [8,6]
        if self.mharp is not None: 
            del self.mharp 
        env = self.prepare_env(problem_config, start1, goal1)
        print "hello ****************************************** "
        print problem_config["robot"]["name"]
        robot = RobotConfig(problem_config["robot"]["name"], problem_config["robot"]["ndofs"], problem_config["robot"]["llimits"], problem_config["robot"]["ulimits"])
        self.mharp = MHARP(env,robot,problem_config)
        return StartServerResponse(1)

if __name__ == "__main__":
    try: 
        rs = ROSService(None,10)
    except Exception,e: 
        print e 
        exit(-1)