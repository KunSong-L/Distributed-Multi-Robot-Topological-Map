#!/usr/bin/python3.8
#多个机器人交汇的各种数据分析
from numpy.lib.function_base import _median_dispatcher
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import csv
import tf
import os
import glob
import re
from scipy.spatial.transform import Rotation as R
import copy
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32
import subprocess
import signal
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from functools import partial
from sensor_msgs.msg import LaserScan, Imu

class change_tf_2_odom:
    def __init__(self, num_robot) -> None:
       
        self.num_robot = num_robot

        self.tf_listener = tf.TransformListener()

        self.odom_pub_list  = [rospy.Publisher(f"robot{i+1}/odom", Odometry, queue_size=100) for i in range(self.num_robot)]
        for i in range(num_robot):
            callback_with_param = partial(self.odom_callback, i)
            rospy.Subscriber(f"robot{i+1}{i+1}{i+1}/odom", Odometry, callback_with_param)
        
        self.scan_pub_list  = [rospy.Publisher(f"robot{i+1}/scan", LaserScan, queue_size=100) for i in range(self.num_robot)]
        for i in range(num_robot):
            #change scan frame
            callback_with_param = partial(self.scan_callback, i)
            rospy.Subscriber(f"robot{i+1}{i+1}{i+1}/scan", LaserScan, callback_with_param)
        
        self.imu_pub_list  = [rospy.Publisher(f"robot{i+1}/flat_imu", Imu, queue_size=100) for i in range(self.num_robot)]
        for i in range(num_robot):
            #change scan frame
            callback_with_param = partial(self.imu_callback, i)
            rospy.Subscriber(f"robot{i+1}{i+1}{i+1}/imu", Imu, callback_with_param)
        

    
    def odom_callback(self,robot_index, msg):
        robot_name = f"robot{robot_index+1}"
        br = tf.TransformBroadcaster()
        translation = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        rotation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        br.sendTransform(translation, rotation, rospy.Time.now(), robot_name +"/base_footprint", robot_name+"/odom")

        msg.header.frame_id = robot_name +"/base_footprint"
        msg.child_frame_id = robot_name+"/odom"
        self.odom_pub_list[robot_index].publish(msg)

    
    def scan_callback(self,robot_index, msg):
        msg.header.frame_id = f"robot{robot_index+1}/base_scan"
        self.scan_pub_list[robot_index].publish(msg)
    
    def imu_callback(self,robot_index, msg):
        msg.header.frame_id = f"robot{robot_index+1}/Imu_link"
        self.imu_pub_list[robot_index].publish(msg)

        
    
if __name__ == '__main__':
    #如果被amcl调用，则同时生成fht map的结果和amcl的结果
    rospy.init_node("odom_2_tf_change")

    num_robot = rospy.get_param('~num_robot')

    
    node = change_tf_2_odom(num_robot)
    rospy.spin()