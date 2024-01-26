#!/usr/bin/python3.8
#记录relocalization用时
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

        self.robot_origin = [rospy.get_param("~origin_x"), rospy.get_param("~origin_y"), rospy.get_param("~origin_yaw")]
        for i in range(3):
            self.robot_origin[i] = float(self.robot_origin[i])
        #计算理论值
        if "museum" in sim_env:
            gt_vector = np.array([7-self.robot_origin[0], 8 - self.robot_origin[1]])
            theta = self.robot_origin[2]
            gt_2 = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).T @ gt_vector
            rot = 90 - np.rad2deg(theta)
            self.gt_rela_pose = [gt_2[0],gt_2[1],rot] #每次需要修改
        if "large_indoor" in sim_env:
            gt_vector = np.array([10-self.robot_origin[0], 10 - self.robot_origin[1]])
            theta = self.robot_origin[2]
            gt_2 = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).T @ gt_vector
            rot = 0 - np.rad2deg(theta)
            self.gt_rela_pose = [gt_2[0],gt_2[1],rot] #每次需要修改

        if reloca_method == "amcl":
            rospy.Subscriber("/tf", TFMessage, self.get_rela_pose_amcl)

        if self.single_robot:
            # 创建CSV文件并写入表头
            with open(path + robot_name + 'topo_reloca' + file_index + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Zeros Count','x/m','y/m','RelaX','RelaY','RelaYaw'])
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
            if self.reloca_method != "amcl":
                rela_pose = self.get_rela_pose()
                if rela_pose != [0,0,0]:
                    self.last_pose = copy.deepcopy(rela_pose)
                else:
                    rela_pose = copy.deepcopy(self.last_pose)
            else:
                rela_pose = self.last_rela_pose_from_tf
            
            for i in range(3):
                rela_pose[i] -= self.gt_rela_pose[i]
            self.map_timestamps.append(map_time)
            self.zeros_counts.append(zeros_count)
            with open(path + robot_name + 'topo_reloca'+file_index+'.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([map_time, zeros_count,now_pose[0],now_pose[1],rela_pose[0],rela_pose[1],rela_pose[2]])
    
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
    
    def get_rela_pose_amcl(self,msg):
        rela_pose = [0,0,0]
        for transform in msg.transforms:
            frame_id = transform.header.frame_id
            child_frame_id = transform.child_frame_id
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            # 在这里可以进行保存或其他处理
            if frame_id == "reloca_map" and child_frame_id == "robot1/odom_amcl":
                rela_pose[0] = translation.x
                rela_pose[1] = translation.y
                rotation_np = np.array([rotation.x,rotation.y,rotation.z,rotation.w])
                rela_pose[2] = R.from_quat(rotation_np).as_euler('xyz', degrees=True)[2]
                self.last_rela_pose_from_tf = copy.deepcopy(rela_pose)

    
    def get_rela_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        rela_pose = [0,0,0]
        try:
            if  self.reloca_method != "amcl":
                self.tf_listener_relo.waitForTransform(robot_name+"/map", robot_name+"/map_origin", tmptimenow, rospy.Duration(0.5))
                tf_transform, rotation = self.tf_listener_relo.lookupTransform(robot_name+"/map", robot_name+"/map_origin", tmptimenow)
                rela_pose[0] = tf_transform[0]
                rela_pose[1] = tf_transform[1]
                rela_pose[2] = R.from_quat(rotation).as_euler('xyz', degrees=True)[2]
            else:
                self.tf_listener_relo.waitForTransform(robot_name+"/odom_amcl", "reloca_map", tmptimenow, rospy.Duration(1))
                tf_transform, rotation = self.tf_listener_relo.lookupTransform(robot_name+"/odom_amcl", "reloca_map")
                rela_pose[0] = tf_transform[0]
                rela_pose[1] = tf_transform[1]
                rela_pose[2] = R.from_quat(rotation).as_euler('xyz', degrees=True)[2]

        except:
            pass

        return rela_pose


if __name__ == '__main__':
    #如果被amcl调用，则同时生成fht map的结果和amcl的结果
    rospy.init_node("map_analysis")

    reloca_method = rospy.get_param('~reloca_method')
    sim_env = rospy.get_param('~sim_env')
    path = "/home/master/FHT_map_data/relocolization/"+reloca_method+"/" + sim_env +"/"
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