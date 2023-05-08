#!/usr/bin/python3.8
from tkinter.constants import Y
import rospy
from rospy.rostime import Duration
from rospy.timer import Rate, sleep
from sensor_msgs.msg import Image
import rospkg
import tf
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from torch import jit
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped, Point
from laser_geometry import LaserProjection
import message_filters
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelStates

from self_topoexplore.msg import UnexploredDirectionsMsg
from self_topoexplore.msg import TopoMapMsg


from TopoMap import Vertex, Edge, TopologicalMap
from utils.imageretrieval.imageretrievalnet import init_network
from utils.imageretrieval.extract_feature import cal_feature
from utils.topomap_bridge import TopomapToMessage, MessageToTopomap

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from queue import Queue
from scipy.spatial.transform import Rotation as R
import math
import time
import copy

from robot_function import *
from RelaPose_2pc_function import *


debug_path = "/home/master/debug/"

class RobotNode:
    def __init__(self, robot_name, robot_list):#输入当前机器人，其他机器人的id list
        

        self.cv_bridge = CvBridge()
        self.panoramic_view_pub = rospy.Publisher(
            robot_name+"/panoramic", Image, queue_size=1)
        self.robot1_ready = 0
        self.robot2_ready = 0
        print(robot_list)
        for robot in robot_list:
            rospy.Subscriber(
                robot+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)



    def create_panoramic_callback(self, image1, image2, image3, image4):
        #合成一张全景图片然后发布
        img1 = self.cv_bridge.imgmsg_to_cv2(image1, desired_encoding="bgr8")
        img2 = self.cv_bridge.imgmsg_to_cv2(image2, desired_encoding="bgr8")
        img3 = self.cv_bridge.imgmsg_to_cv2(image3, desired_encoding="bgr8")
        img4 = self.cv_bridge.imgmsg_to_cv2(image4, desired_encoding="bgr8")
        panoram = [img1, img2, img3, img4]
        self.panoramic_view = np.hstack(panoram)
        # cv2.imwrite("/home/master/debug/panormaic.jpg", cv2.cvtColor(self.panoramic_view, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/1.jpg", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/2.jpg", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/3.jpg", cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/4.jpg", cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
        image_message = self.cv_bridge.cv2_to_imgmsg(self.panoramic_view, encoding="bgr8")
        image_message.header.stamp = rospy.Time.now()  
        image_message.header.frame_id = robot_name+"/odom"
        self.panoramic_view_pub.publish(image_message)

    def map_panoramic_callback(self, panoramic):
        if panoramic.header.frame_id == "robot1/odom":
            panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="bgr8")
            cv2.imwrite(debug_path+"robot1.jpg", panoramic_view)
            self.robot1_ready=1
        if panoramic.header.frame_id == "robot2/odom":
            panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="bgr8")
            cv2.imwrite(debug_path+"robot2.jpg", panoramic_view)
            self.robot2_ready=1
        
        if self.robot1_ready == 0 or self.robot2_ready==0:
            return
        
        #init down
        



if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('topological_map')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    print(robot_name, robot_num)

    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    
    node = RobotNode(robot_name, robot_list)

    print("node init done")
    #订阅自己的图像
    robot1_image1_sub = message_filters.Subscriber(robot_name+"/camera1/image_raw", Image)
    robot1_image2_sub = message_filters.Subscriber(robot_name+"/camera2/image_raw", Image)
    robot1_image3_sub = message_filters.Subscriber(robot_name+"/camera3/image_raw", Image)
    robot1_image4_sub = message_filters.Subscriber(robot_name+"/camera4/image_raw", Image)
    ts = message_filters.TimeSynchronizer([robot1_image1_sub, robot1_image2_sub, robot1_image3_sub, robot1_image4_sub], 10) #传感器信息融合
    ts.registerCallback(node.create_panoramic_callback) # 

    rospy.spin()