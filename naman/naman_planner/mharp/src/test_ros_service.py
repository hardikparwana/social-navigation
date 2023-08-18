#! /usr/bin/python 

import rospy
from mharp_msgs.srv import StartServer, GetMHARPMotionPlan

if __name__ == "__main__":
    rospy.wait_for_service("start_motion_planning_server")
    try: 
        start_server_proxy = rospy.ServiceProxy("start_motion_planning_server", StartServer)
        response = start_server_proxy("parameters.yaml")
        print response
    except rospy.ServiceException, e: 
        print e    
    rospy.wait_for_service("/get_mharm_motion_plan")
    # exit()
    try: 
        mp_proxy = rospy.ServiceProxy("get_mharm_motion_plan", GetMHARPMotionPlan)
        response = mp_proxy(1,[-8,-8,0],[-8,-6,0])
        print response
    except rospy.ServiceException,e: 
        print e