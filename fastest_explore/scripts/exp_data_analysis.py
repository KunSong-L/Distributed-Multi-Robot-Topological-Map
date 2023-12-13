#!/usr/bin/python3.8
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


class data_analysis:
    def __init__(self, robot_name) -> None:
       
        self.self_robot_name = robot_name
        # self.pose_pub = rospy.Publisher(
        #     robot_name+"/testpose", PoseStamped, queue_size=10)
        self.map_timestamps = []
        self.zeros_counts = []
        self.single_robot = 1
        self.tf_listener = tf.TransformListener()
        if self.single_robot:
            # 创建CSV文件并写入表头
            with open(path + robot_name + 'map_complete' + file_index + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Zeros Count','x/m','y/m'])
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_callback, queue_size=1)
    
    def map_callback(self, map):
        # print(map.info.origin.position)
        map_message = OccupancyGrid()
        map_message.header = map.header
        map_message.info = map.info
        # print("map orientation::", map.info.origin)
        shape = (map.info.height, map.info.width)
        mapdata = np.asarray(map.data).reshape(shape)
        if self.single_robot:
            # Count the number of zeros in the map
            zeros_count = np.sum(mapdata == 0)
            # Save the map timestamp and number of zeros in a file
            map_time = map.header.stamp.to_sec()
            now_pose = self.update_robot_pose()
            self.map_timestamps.append(map_time)
            self.zeros_counts.append(zeros_count)
            with open(path + robot_name + 'map_complete'+file_index+'.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([map_time, zeros_count,now_pose[0],now_pose[1]])
    
    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        self.tf_listener.waitForTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
        pose = [0,0]
        try:
            tf_transform, rotation = self.tf_listener.lookupTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow)
            pose[0] = tf_transform[0]
            pose[1] = tf_transform[1]

        except:
            pass

        return pose


if __name__ == '__main__':
    rospy.init_node("data_analysis")
    path = rospy.get_param("~file_path")
    file_paths = glob.glob(os.path.join(path, "*"))

    # 按文件名进行排序
    sorted_file_paths = sorted(file_paths, key=lambda x: os.path.basename(x))

    # 使用正则表达式提取所有数字
    if len(sorted_file_paths)==0:
        file_index = "1"
    else:
        numbers = re.findall(r"\d+", sorted_file_paths[-1])
        numbers = [int(number) for number in numbers]
        file_index = str(max(numbers)+1)

    robot_name = rospy.get_param("~robot_name")
    node = data_analysis(robot_name)
    rospy.spin()