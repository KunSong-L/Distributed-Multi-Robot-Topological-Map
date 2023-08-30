#!/usr/bin/python3.8
#记录relocalization用时
from numpy.lib.function_base import _median_dispatcher
import rospy
from sensor_msgs.msg import Image, LaserScan,PointCloud2, PointField
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

class map_analysis:
    def __init__(self, robot_name,reloca_method) -> None:
        print(robot_name)
        
        self.self_robot_name = robot_name
        # self.pose_pub = rospy.Publisher(
        #     robot_name+"/testpose", PoseStamped, queue_size=10)
        self.map_timestamps = []
        self.zeros_counts = []
        self.single_robot = 1
        self.tf_listener = tf.TransformListener()
        self.tf_listener_relo = tf.TransformListener()
        self.last_pose = [0,0,0]
        self.reloca_method = reloca_method
        self.last_rela_pose_from_tf = [0,0,0]
        self.gt_rela_pose = [7,0,90]
        #目标值
        self.nav_target = [rospy.get_param("~target_x"), rospy.get_param("~target_y"), rospy.get_param("~target_yaw")]
        for i in range(3):
            self.nav_target[i] = float(self.nav_target[i])
        nav_target_x = self.nav_target[1] - 8
        nav_target_y = 7 - self.nav_target[0]
        self.target_map_frame = [nav_target_x,nav_target_y,0]

        if self.single_robot:
            # 创建CSV文件并写入表头
            with open(path + robot_name + 'topo_reloca' + file_index + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Zeros Count','x/m','y/m'])
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_callback, queue_size=1)

    
    def map_callback(self, data):
        # print(map.info.origin.position)

        if self.single_robot:
            # Count the number of zeros in the map
            # Save the map timestamp and number of zeros in a file
            now_pose = self.update_robot_pose()
        
            with open(path + robot_name + 'topo_reloca'+file_index+'.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([data.header.stamp.to_sec(), 0,now_pose[0] - self.target_map_frame[0],now_pose[1]- self.target_map_frame[1]])
    
    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        pose = [0,0]
        self.tf_listener.waitForTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
        try:
            tf_transform, rotation = self.tf_listener.lookupTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow)
            pose[0] = tf_transform[0]
            pose[1] = tf_transform[1]

        except:
            pass

        return pose
    


if __name__ == '__main__':
    #如果被amcl调用，则同时生成fht map的结果和amcl的结果
    rospy.init_node("map_analysis")

    reloca_method = rospy.get_param('~nav_method')
    sim_env = rospy.get_param('~sim_env')
    path = "/home/master/topomap_data/navigation/"+reloca_method+"/" + sim_env +"/"
    file_paths = glob.glob(os.path.join(path, "*"))

    # 按文件名进行排序
    sorted_file_paths = sorted(file_paths, key=lambda x: os.path.basename(x))

    # 使用正则表达式提取所有数字
    if len(sorted_file_paths)==0:
        file_index = "1"
    else:
        numbers = re.findall(r"\d+", sorted_file_paths[-1])
        numbers = [int(number) for number in numbers]
        file_index = str(len(sorted_file_paths) + 1)

    robot_name = "robot1"
    node = map_analysis(robot_name,reloca_method)
    rospy.spin()