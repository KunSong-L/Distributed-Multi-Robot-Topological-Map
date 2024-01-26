#!/usr/bin/python3.8
#创建一张静止的fhtmap
from tkinter.constants import Y
import rospy
from sensor_msgs.msg import Image, LaserScan
import rospkg
import tf
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from laser_geometry import LaserProjection
import message_filters
from self_topoexplore.msg import TopoMapMsg
from TopoMap import Support_Vertex, Vertex, Edge, TopologicalMap
from utils.imageretrieval.imageretrievalnet import init_network
from utils.pointnet_model import PointNet_est
from utils.imageretrieval.extract_feature import cal_feature
from utils.topomap_bridge import TopomapToMessage
from utils.astar import grid_path, topo_map_path
import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import copy
from robot_function import *
import sys
import scipy.ndimage
from nav_msgs.msg import OccupancyGrid



debug_path = "/home/master/debug/test1/"
save_result = False

class static_fht_map:
    def __init__(self, robot_name,topomap):#输入当前机器人，其他机器人的id list

        #color: frontier, main_vertex, support vertex, edge, local free space
        robot1_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF],[0x00, 0xFF, 0x00]])/255.0
        robot2_color = np.array([[0xFF, 0xA5, 0x00], [0xDC, 0x14, 0xb1], [0x16, 0x7c, 0xdf], [0x00, 0x64, 0x00], [0x40, 0xE0, 0xD0],[0xb5, 0xbc, 0x38]]) / 255.0
        robot3_color = np.array([[0x8A, 0x2B, 0xE2], [0x8B, 0x00, 0x00], [0xFF, 0xF8, 0xDC], [0x7B, 0x68, 0xEE], [0xFF, 0x45, 0x00],[0xF0, 0xF8, 0xFF]]) / 255.0
        self.vis_color = [robot1_color,robot2_color,robot3_color]
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.map = topomap
        self.adj_list = dict()
        self.edge_to_adj_list()
        self.self_robot_name = "robot1" #全局地图
        self.robot_index = 0


        #publisher and subscriber
        self.marker_pub = rospy.Publisher(robot_name+"/visualization/marker", MarkerArray, queue_size=1)
        self.edge_pub = rospy.Publisher(robot_name+"/visualization/edge", MarkerArray, queue_size=1)
        self.vertex_free_space_pub = rospy.Publisher(robot_name+'/vertex_free_space', MarkerArray, queue_size=1)


    def visulize_vertex(self):
        #可视化vertex free space
        # 创建所有平面的Marker消息
        markers = []
        markerid = 0
        
        now_map = self.map
        ori_map = R.from_matrix(now_map.rotation).as_quat()
        now_vis_color = self.vis_color[self.robot_index]
        for now_vertex in now_map.vertex:
            if now_vertex.local_free_space_rect == [0,0,0,0]:
                continue
            x1,y1,x2,y2 = now_vertex.local_free_space_rect
            marker = Marker()
            marker.header.frame_id = self.self_robot_name + "/map"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (x1 + x2)/2.0
            marker.pose.position.y = (y1 + y2)/2.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = ori_map[0]
            marker.pose.orientation.y = ori_map[1]
            marker.pose.orientation.z = ori_map[2]
            marker.pose.orientation.w = ori_map[3]
            tmp = now_map.rotation.T @ np.array([x2-x1,y2-y1,0])
            
            marker.scale.x = abs(tmp[0])
            marker.scale.y = abs(tmp[1])
            marker.scale.z = 0.03 # 指定平面的厚度
            marker.color.r = now_vis_color[5][0]
            marker.color.g = now_vis_color[5][1]
            marker.color.b = now_vis_color[5][2]
            marker.color.a = 0.2 # 指定平面的透明度
            marker.id = markerid
            markers.append(marker)
            markerid +=1
        # 将所有Marker消息放入一个MarkerArray消息中，并发布它们
        marker_array = MarkerArray()
        marker_array.markers = markers
        self.vertex_free_space_pub.publish(marker_array)
        
        #可视化vertex
        marker_array = MarkerArray()
        markerid = 0
        main_vertex_color = (now_vis_color[1][0], now_vis_color[1][1], now_vis_color[1][2])
        support_vertex_color = (now_vis_color[2][0], now_vis_color[2][1], now_vis_color[2][2])
        for vertex in now_map.vertex:
            if isinstance(vertex, Vertex):
                marker_message = set_marker(self.self_robot_name, markerid, vertex.pose, color=main_vertex_color, scale=0.5)
            else:
                marker_message = set_marker(self.self_robot_name, markerid, vertex.pose, color=support_vertex_color, scale=0.4)
            
            marker_array.markers.append(marker_message)
            markerid += 1
        
        #visualize edge
        #可视化edge就是把两个vertex的pose做一个连线
        
        edge_array = MarkerArray()
        now_edge_id = 0
        main_edge_color = (now_vis_color[3][0], now_vis_color[3][1], now_vis_color[3][2])

        for edge in now_map.edge:
            poses = []
            poses.append(now_map.vertex[edge.link[0][1]].pose)
            poses.append(now_map.vertex[edge.link[1][1]].pose)
            edge_message = set_edge(self.self_robot_name, now_edge_id, poses, "edge",main_edge_color, scale=0.1,frame_name = "/map")
            now_edge_id += 1
            edge_array.markers.append(edge_message)
        
        self.marker_pub.publish(marker_array)
        self.edge_pub.publish(edge_array)

    def topo_path_planning(self,point1,point2,consider_ori = True):
        #输入两个点: point1/2: 格式可能为[x,y,yaw]或者[x,y]
        #输出两个点之间规划的一条轨迹和轨迹的时间代价
        #在这个函数中，考虑了机器人转向的代价
        if len(self.map.vertex) == 0:
            return np.linalg.norm(point1[0:2] - point2[0:2]),[point1,point2] #TODO
        
        point1_pixel = (np.array(point1[0:2])- np.array(self.map_origin))/self.map_resolution
        point2_pixel = (np.array(point2[0:2])- np.array(self.map_origin))/self.map_resolution
        if self.free_space_line(point1_pixel,point2_pixel):
            return self.path_distance_cost(point1,point2,[]),[point1,point2]

        start_in_vertex_index = self.dual_vertex_of_a_point(point1)
        target_in_vertex_index = self.dual_vertex_of_a_point(point2)
        
        
        #get total id and pose
        shortest_path_length = 1e100
        target_path = None
        start_point_pose = np.array([point1[0],point1[1]])
        end_point_pose = np.array([point2[0],point2[1]])
        for now_start in start_in_vertex_index:
            now_start_pose = np.array(self.map.vertex[now_start].pose[0:2])
            for now_end in target_in_vertex_index:
                now_end_pose = np.array(self.map.vertex[now_end].pose[0:2])
                target_id_list = [now_start, now_end]
                topo_map = topo_map_path(self.adj_list,target_id_list[-1], target_id_list[0:-1])
                topo_map.get_path()
                now_path_length = topo_map.path_length[0] + np.linalg.norm(start_point_pose - now_start_pose) + np.linalg.norm(end_point_pose - now_end_pose)
                if now_path_length < shortest_path_length:
                    shortest_path_length = now_path_length
                    target_path = copy.deepcopy(topo_map.foundPath[0][::-1])
        
        # target_path: 从start index到end index的一个list
        # print(target_path)
        path_list = []
        for now_path_index in target_path:
            path_list.append(self.map.vertex[now_path_index].pose[0:2])
        
        #计算总的运动代价
        if consider_ori: #如果考虑旋转代价
            path_length = self.path_distance_cost(point1,point2,path_list)
        else: #不考虑
            tmp = [point1] + path_list + [point2]
            l = 0
            for i in range(len(tmp) - 1):
                a1 = np.array(tmp[i])
                a2 = np.array(tmp[i+1])
                l += np.linalg.norm(a1-a2)
            path_length = l/self.max_v

        return path_length, [point1] + path_list + [point2]


    def edge_to_adj_list(self):
        for now_edge in self.map.edge:
            first_id = now_edge.link[0][1]
            last_id = now_edge.link[1][1]
            pose1 = self.map.vertex[first_id].pose[0:2]
            pose2 = self.map.vertex[last_id].pose[0:2]
            if first_id not in self.adj_list.keys():
                self.adj_list[first_id]  = []
            if last_id not in self.adj_list.keys():
                self.adj_list[last_id]  = []
            
            cost = ((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)**0.5
            self.adj_list[first_id].append((last_id, cost))
            self.adj_list[last_id].append((first_id, cost))        

    
    def free_space_line_map(self,point1,point2,now_global_map):
        # check whether a line cross a free space
        # point1 and point2 in pixel frame
        height, width = now_global_map.shape

        x1, y1 = point1
        x2, y2 = point2

        distance = max(abs(x2 - x1), abs(y2 - y1))

        step_x = (x2 - x1) / distance
        step_y = (y2 - y1) / distance

        for i in range(int(distance) + 1):
            x = int(x1 + i * step_x)
            y = int(y1 + i * step_y)
            if x < 0 or x >= width or y < 0 or y >= height or now_global_map[y, x] != 0:
                # if now_global_map[y, x] != 255:#排除掉经过unknown的部分
                return False
        return True

    def free_space_line(self,point1,point2):
        # check whether a line cross a free space
        # (这里把unknown也作为free space，主要是考虑了地图更新速度比机器人慢，机器人容易驶入unkown free space)
        # point1 and point2 in pixel frame
        now_global_map = self.global_map
        height, width = now_global_map.shape

        x1, y1 = point1
        x2, y2 = point2

        distance = max(abs(x2 - x1), abs(y2 - y1))
        if distance==0:
            return False
        step_x = (x2 - x1) / distance
        step_y = (y2 - y1) / distance

        for i in range(int(distance) + 1):
            x = int(x1 + i * step_x)
            y = int(y1 + i * step_y)
            if x < 0 or x >= width or y < 0 or y >= height or now_global_map[y, x] != 0:
                if now_global_map[y, x] != 255:#排除掉经过unknown的部分
                    return False
        return True

    def expanded_free_space_line(self,point1,point2, offset_length):
        x1, y1 = point1
        x2, y2 = point2
        vertical_vector = np.array([y2 - y1, x1 - x2])
        vertical_vector = vertical_vector/ np.linalg.norm(vertical_vector)
        offset_points1 = [point1 + vertical_vector*offset_length, point2 + vertical_vector*offset_length]
        offset_points2 = [point1 - vertical_vector*offset_length, point2 - vertical_vector*offset_length]

        return self.free_space_line(point1, point2) and self.free_space_line(offset_points1[0], offset_points1[1]) and self.free_space_line(offset_points2[0], offset_points2[1])

    def dual_vertex_of_a_point(self,point):
        #给定一个地图上的点，返回空白区域包含了这个点的所有vertex
        #如果上述条件不存在，则返回一个最近的点
        start_x = point[0]
        start_y = point[1]
        start_in_vertex_index = []
        for i in range(len(self.map.vertex)):
            now_free_space = self.map.vertex[i].local_free_space_rect
            if now_free_space[0] < start_x and start_x < now_free_space[2] and now_free_space[1] < start_y and start_y < now_free_space[3]:
                start_in_vertex_index.append(i)
        
        if len(start_in_vertex_index) == 0:
            #对于目标点不在free space 情况，找一个最近的vertex导航过去
            min_dis = 1e100
            min_index = -1
            for i in range(len(self.map.vertex)):
                now_vertex_pose = self.map.vertex[i].pose[0:2]
                now_dis = ((start_x - now_vertex_pose[0])**2 + (start_y - now_vertex_pose[1])**2)**0.5
                if now_dis < min_dis:
                    min_dis = now_dis
                    min_index = i
            start_in_vertex_index.append(min_index)

        return start_in_vertex_index


    

    def path_distance_cost(self,point1,point2,path_point):
        #point1, point2: 两个以[x,y,yaw]格式给出的点
        #path_points: 一系列以[x,y]格式给出的点所组成的list
        delete_first = False
        delete_last = False
        if len(point1) == 2:#如果某一个点是二维的，就删去对应的角度代价
            point1 = [point1[0],point1[1],0]
            delete_first = True
        if len(point2) == 2:
            point2 = [point2[0],point2[1],0]
            delete_last = True
        
        now_dir = point1[2]
        now_pose = point1[0:2]
        total_cost = 0
        rot_cost_list = []
        for index in range(len(path_point)):
            #前往下标为index的点
            #代价为先再前进
            next_pose = path_point[index]
            move_cost = np.linalg.norm(np.array(now_pose) - np.array(next_pose))/self.max_v

            next_dir = np.arctan2(next_pose[1] - now_pose[1], next_pose[0] - now_pose[0])

            rot_theta = next_dir - now_dir
            rot_cost = np.abs(np.arctan2(np.sin(rot_theta),np.cos(rot_theta)))/self.max_w
            
            total_cost = total_cost + rot_cost + move_cost

            now_dir = next_dir
            now_pose = next_pose

            rot_cost_list.append(rot_cost)
        
        #此时到达路径上最后一个点，需要前进到最后一个点再转向

        next_pose = point2[0:2]
        next_dir = np.arctan2(next_pose[1] - now_pose[1], next_pose[0] - now_pose[0])
        move_cost = np.linalg.norm(np.array(now_pose) - np.array(next_pose))/self.max_v
        rot_theta = next_dir - now_dir
        now_dir = next_dir
        rot_cost = np.abs(np.arctan2(np.sin(rot_theta),np.cos(rot_theta)))/self.max_w
        total_cost = total_cost + rot_cost + move_cost
        rot_cost_list.append(rot_cost)

        #最后再旋转一下
        next_dir = point2[2]
        rot_theta = next_dir - now_dir
        rot_cost = np.abs(np.arctan2(np.sin(rot_theta),np.cos(rot_theta)))/self.max_w
        total_cost = total_cost + rot_cost

        if delete_first:
            total_cost-=rot_cost_list[0]
        if delete_last:
            total_cost-=rot_cost_list[-1]
        return total_cost
