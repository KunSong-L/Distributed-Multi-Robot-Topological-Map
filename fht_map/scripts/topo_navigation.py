#!/usr/bin/python3.8
from tkinter.constants import Y
import rospy
from rospy.rostime import Duration
from rospy.timer import Rate, sleep
from sensor_msgs.msg import Image, LaserScan,PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import rospkg
import tf
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from torch import jit
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped, Point, TransformStamped
from laser_geometry import LaserProjection
import message_filters
from utils.astar import grid_path, topo_map_path
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelStates
from self_topoexplore.msg import UnexploredDirectionsMsg
from self_topoexplore.msg import TopoMapMsg
from self_topoexplore.msg import ImageWithPointCloudMsg
from TopoMap import Vertex, Edge, TopologicalMap, Support_Vertex
from utils.imageretrieval.imageretrievalnet import init_network
from utils.imageretrieval.extract_feature import cal_feature
from utils.topomap_bridge import TopomapToMessage, MessageToTopomap
from utils.ColorMapping import *
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
from std_msgs.msg import Header
import sys
from robot_function import *
from RelaPose_2pc_function import *
import tf2_ros
import subprocess
from ransac_icp import *
import open3d as o3d

debug_path = "/home/master/debug/test1/"
save_result = False

class RobotNode:
    def __init__(self, robot_name, robot_list):#输入当前机器人，其他机器人的id list
        rospack = rospkg.RosPack()
        self.self_robot_name = robot_name
        path = rospack.get_path('self_topoexplore')
        # in simulation environment each robot has same intrinsic matrix
        self.K_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))
        #network part
        network = rospy.get_param("~network")
        self.network_gpu = rospy.get_param("~platform")
        if network in PRETRAINED:
            state = load_url(PRETRAINED[network], model_dir= os.path.join(path, "data/networks"))
        else:
            state = torch.load(network)
        net_params = get_net_param(state)
        torch.cuda.empty_cache()
        self.net = init_network(net_params)
        self.net.load_state_dict(state['state_dict'])
        self.net.cuda(self.network_gpu)
        self.net.eval()
        self.test_index = 0
        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        #robot data
        self.pose = [0,0,0] # x y yaw angle in degree
        self.init_map_angle_ready = 0
        self.map_orientation = None
        self.map_angle = None #Yaw angle of map
        self.current_loc_pixel = [0,0]
        self.erro_count = 0
        self.goal = np.array([])
        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x00, 0x96, 0xC7]])/255.0
        self.laserProjection = LaserProjection()
        self.pcd_queue = Queue(maxsize=10)# no used
        self.grid_map_ready = 0
        self.tf_transform_ready = 0
        self.cv_bridge = CvBridge()
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.received_map = None #original topomap
        self.last_vertex = -1
        self.current_node = None
        self.last_nextmove = 0 #angle
        self.topomap_meet = 0
        self.vertex_map_ready = False
        self.vertex_dict = dict()
        self.vertex_dict[self.self_robot_name] = list()
        self.matched_vertex_dict = dict()
        self.matched_vertex = [] #already estimated relative pose
        self.now_feature = np.array([])

        self.edge_dict = dict()
        self.relative_position = dict()
        self.relative_orientation = dict()
        self.meeted_robot = ["robot2"]
        for item in robot_list:
            self.vertex_dict[item] = list()
            self.matched_vertex_dict[item] = list()
            self.edge_dict[item] = list()
            self.relative_position[item] = [0, 0, 0]
            self.relative_orientation[item] = 0


        self.topomap_matched_score = 0.7
        self.receive_topomap = False
        self.init_map_pose = True
        self.navigated_point = np.array([],dtype=float).reshape((-1,3))
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        #navigation
        #获取相对位姿，直接完成重定位
              
        try:
            now_env = rospy.get_param("~sim_env")
        except:
            print("no simulation env input!-----------------\n\n\n")
            now_env = "museum"
        
        if now_env == "large_indoor":
            self.world_map1 = [10,10,0]
        elif now_env == "museum":
            self.world_map1 = [7,8,1.57]
        elif now_env == "large_2_2": 
            self.world_map1 = [5,5,1.57]
        elif  now_env == "large_2_4":
            self.world_map1 = [5,50,1.57]
        
        self.robot_origin = [rospy.get_param("~origin_x"), rospy.get_param("~origin_y"), rospy.get_param("~origin_yaw")]
        for i in range(3):
            self.robot_origin[i] = float(self.robot_origin[i])
        #计算理论值
        gt_vector = np.array([self.world_map1[0]-self.robot_origin[0], self.world_map1[1] - self.robot_origin[1]])
        theta = self.robot_origin[2]
        gt_2 = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).T @ gt_vector
        rot = np.rad2deg(self.world_map1[2]) - np.rad2deg(theta)
        self.map2_map1 = [gt_2[0],gt_2[1],np.deg2rad(rot)]#原始map在新map坐标系下位置
        self.map1_map2 = change_frame([0,0,0], self.map2_map1) 

        self.world_map2 = [rospy.get_param("~origin_x"), rospy.get_param("~origin_y"), rospy.get_param("~origin_yaw")]
        #获取目标点
        self.nav_target = [rospy.get_param("~target_x"), rospy.get_param("~target_y"), rospy.get_param("~target_yaw")]
        for i in range(3):
            self.nav_target[i] = float(self.nav_target[i])
        self.update_navigation_path_flag = False
        self.target_topo_path = []
        self.now_target_vertex = 0 #目前机器人所需要去的下标
        self.target_on_freespace = False
        self.target_map_frame = [0,0,0]

        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        self.laser_scan_cos_sin = None
        self.laser_scan_init = False
        self.local_laserscan = None
        self.local_laserscan_angle = None
        #move base
        self.actoinclient = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)
        self.trajectory_rate = Rate(0.3)
        self.trajectory_length = 0
        self.start_time = time.time()
        self.total_frontier = np.array([],dtype=float).reshape(-1,2)

        

        #publisher and subscriber
        self.marker_pub = rospy.Publisher(
            robot_name+"/visualization/marker", MarkerArray, queue_size=1)
        self.edge_pub = rospy.Publisher(
            robot_name+"/visualization/edge", MarkerArray, queue_size=1)
        self.twist_pub = rospy.Publisher(
            robot_name+"/mobile_base/commands/velocity", Twist, queue_size=1)
        self.goal_pub = rospy.Publisher(
            robot_name+"/goal", PoseStamped, queue_size=1)
        self.panoramic_view_pub = rospy.Publisher(
            robot_name+"/panoramic", Image, queue_size=1)
        self.topomap_pub = rospy.Publisher(
            robot_name+"/topomap", TopoMapMsg, queue_size=1)
        self.unexplore_direction_pub = rospy.Publisher(
            robot_name+"/ud", UnexploredDirectionsMsg, queue_size=1)
        self.start_pub = rospy.Publisher(
            "/start_exp", String, queue_size=1) #发一个start
        self.pc_pub = rospy.Publisher(robot_name+'/point_cloud', PointCloud2, queue_size=10)
        self.vertex_free_space_pub = rospy.Publisher(robot_name+'/vertex_free_space', MarkerArray, queue_size=1)

        self.frontier_publisher = rospy.Publisher(robot_name+'/frontier_points', Marker, queue_size=1)
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/topomap", TopoMapMsg, self.topomap_callback, queue_size=1, buff_size=52428800)
        rospy.Subscriber(
            robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/scan", LaserScan, self.laserscan_callback, queue_size=1)
        self.actoinclient.wait_for_server()


    def laserscan_callback(self, scan):
        ranges = np.array(scan.ranges)
        
        if not self.laser_scan_init:
            angle_min = scan.angle_min
            angle_increment = scan.angle_increment
            laser_cos = np.cos(angle_min + angle_increment * np.arange(len(ranges)))
            laser_sin = np.sin(angle_min + angle_increment * np.arange(len(ranges)))
            self.laser_scan_cos_sin = np.stack([laser_cos, laser_sin])
            self.laser_scan_init = True
        
        valid_indices = np.isfinite(ranges)
        self.local_laserscan  = np.array(ranges * self.laser_scan_cos_sin)[:,valid_indices]
        self.local_laserscan_angle = ranges



    def create_panoramic_callback(self, image1, image2, image3, image4):
        #合成一张全景图片然后发布
        img1 = self.cv_bridge.imgmsg_to_cv2(image1, desired_encoding="rgb8")
        img2 = self.cv_bridge.imgmsg_to_cv2(image2, desired_encoding="rgb8")
        img3 = self.cv_bridge.imgmsg_to_cv2(image3, desired_encoding="rgb8")
        img4 = self.cv_bridge.imgmsg_to_cv2(image4, desired_encoding="rgb8")
        panoram = [img1, img2, img3, img4]
        self.panoramic_view = np.hstack(panoram)
        image_message = self.cv_bridge.cv2_to_imgmsg(self.panoramic_view, encoding="rgb8")
        image_message.header.stamp = rospy.Time.now()  
        image_message.header.frame_id = robot_name+"/odom"
        self.panoramic_view_pub.publish(image_message)

      

    def visulize_vertex(self):
        # ----------visualize frontier------------
        
        frontier_marker = Marker()
        now = rospy.Time.now()
        frontier_marker.header.frame_id = robot_name + "/map"
        frontier_marker.header.stamp = now
        frontier_marker.ns = "frontier_point"
        frontier_marker.type = Marker.POINTS
        frontier_marker.action = Marker.ADD
        frontier_marker.pose.orientation.w = 1.0
        frontier_marker.scale.x = 0.1
        frontier_marker.scale.y = 0.1
        frontier_marker.color.r = self.vis_color[0][0]
        frontier_marker.color.g = self.vis_color[0][1]
        frontier_marker.color.b = self.vis_color[0][2]
        frontier_marker.color.a = 0.7
        for frontier in self.total_frontier:
            point_msg = Point()
            point_msg.x = frontier[0]
            point_msg.y = frontier[1]
            point_msg.z = 0.2
            frontier_marker.points.append(point_msg)
        self.frontier_publisher.publish(frontier_marker)
        # --------------finish visualize frontier---------------

        #可视化vertex free space
        # 创建所有平面的Marker消息
        markers = []
        for index, now_vertex in enumerate(self.map.vertex):
            if now_vertex.local_free_space_rect == [0,0,0,0]:
                continue
            x1,y1,x2,y2 = now_vertex.local_free_space_rect
            marker = Marker()
            marker.header.frame_id = robot_name + "/map_origin"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (x1 + x2)/2.0
            marker.pose.position.y = (y1 + y2)/2.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = abs(x2 - x1)
            marker.scale.y = abs(y2 - y1)
            marker.scale.z = 0.03 # 指定平面的厚度
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.2 # 指定平面的透明度
            marker.id = index
            markers.append(marker)
        # 将所有Marker消息放入一个MarkerArray消息中，并发布它们
        marker_array = MarkerArray()
        marker_array.markers = markers
        self.vertex_free_space_pub.publish(marker_array)
        
        #可视化vertex
        # marker_array = MarkerArray()
        # marker_message = set_marker(robot_name, len(self.map.vertex), self.map.vertex[0].pose, action=Marker.DELETEALL,frame_name = "/map_origin")
        # marker_array.markers.append(marker_message)
        # self.marker_pub.publish(marker_array) #DELETEALL 操作，防止重影
        marker_array = MarkerArray()
        markerid = 0
        main_vertex_color = (self.vis_color[1][0], self.vis_color[1][1], self.vis_color[1][2])
        support_vertex_color = (self.vis_color[2][0], self.vis_color[2][1], self.vis_color[2][2])

        for vertex in self.map.vertex:
            if vertex.robot_name != robot_name:
                marker_message = set_marker(robot_name, markerid, vertex.pose)#other color
            else:
                if isinstance(vertex, Vertex):
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=main_vertex_color, scale=0.3,frame_name = "/map_origin")
                else:
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=support_vertex_color, scale=0.25,frame_name = "/map_origin")
            marker_array.markers.append(marker_message)
            markerid += 1
        
        #visualize edge
        #可视化edge就是把两个vertex的pose做一个连线
        main_edge_color = (self.vis_color[3][0], self.vis_color[3][1], self.vis_color[3][2])
        edge_array = MarkerArray()
        for edge in self.map.edge:
            num_count = 0
            poses = []
            for vertex in self.map.vertex:
                # find match
                if (edge.link[0][0]==vertex.robot_name and edge.link[0][1]==vertex.id) or (edge.link[1][0]==vertex.robot_name and edge.link[1][1]==vertex.id):
                    poses.append(vertex.pose)
                    num_count += 1
                if num_count == 2:
                    edge_message = set_edge(robot_name, edge.id, poses, "edge",main_edge_color, scale=0.1,frame_name = "/map_origin")
                    edge_array.markers.append(edge_message)
                    break
        self.marker_pub.publish(marker_array)
        self.edge_pub.publish(edge_array)

    def choose_nav_goal(self):
        dis_epos = 1
        angle_epos = 2
        
        frontier_poses = self.total_frontier  

        dis_frontier_poses = np.sqrt(np.sum(np.square(frontier_poses - self.pose[0:2]), axis=1))
        dis_tmp = np.exp(-(dis_frontier_poses - 5.6)**2 / 8)

        angle_frontier_poses = np.arctan2(frontier_poses[:, 1] - self.pose[1], frontier_poses[:, 0] - self.pose[0]) - self.pose[2] / 180 * np.pi
        angle_tmp = np.exp(-angle_frontier_poses**2 / 1)

        frontier_scores = dis_epos * dis_tmp + angle_epos * angle_tmp
        max_index = np.argmax(frontier_scores)

        return self.total_frontier[max_index]

    
    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        self.tf_listener2.waitForTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
        try:
            self.tf_transform, self.rotation = self.tf_listener2.lookupTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow)
            self.tf_transform_ready = 1
            self.pose[0] = self.tf_transform[0]
            self.pose[1] = self.tf_transform[1]
            self.pose[2] = R.from_quat(self.rotation).as_euler('xyz', degrees=True)[2]

            if self.init_map_angle_ready == 0:
                self.map_angle = self.pose[2]
                self.map.offset_angle = self.map_angle
                self.init_map_angle_ready = 1
        except:
            pass

    def angle_laser_to_xy(self, laser_angle):
        # input: laser_angle : 1*n array
        # output: laser_xy : 2*m array with no nan
        angle_min = 0
        angle_increment = 0.017501922324299812
        laser_cos = np.cos(angle_min + angle_increment * np.arange(len(laser_angle)))
        laser_sin = np.sin(angle_min + angle_increment * np.arange(len(laser_angle)))
        laser_scan_cos_sin = np.stack([laser_cos, laser_sin])
        valid_indices = np.isfinite(laser_angle)
        laser_xy  = np.array(laser_angle * laser_scan_cos_sin)[:,valid_indices]
        return laser_xy

    def update_relative_pose(self):
        # 定义转换关系的消息

        relative_pose= self.map2_map1
        orientation = R.from_euler('z', relative_pose[2], degrees=False).as_quat()

        transform = TransformStamped()
        transform.header.frame_id = self.self_robot_name + '/map'  # 源坐标系
        transform.child_frame_id = self.self_robot_name + '/map_origin'  # 目标坐标系
        
        # 设置转换关系的平移部分
        transform.transform.translation.x = relative_pose[0] # 在X轴上的平移量
        transform.transform.translation.y = relative_pose[1] # 在Y轴上的平移量
        transform.transform.translation.z = 0.0  # 在Z轴上的平移量
        
        # 设置转换关系的旋转部分（四元数表示）
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        transform.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(transform)

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

    def map_panoramic_callback(self, panoramic):
        # for navigation
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        self.update_robot_pose() #update robot pose
        self.update_relative_pose()
        if not self.receive_topomap:
            return
        
        if not self.update_navigation_path_flag:
            #如果收到拓扑地图开始导航
            print("-----------Start Nav-------------")
            
            self.vertex_map_ready = True
            #start navigation
            #target original in wrold frame, so change into map frame
            self.map2_target = change_frame(self.nav_target, self.world_map2) #目标在当前地图坐标系中位置
            self.map1_target = change_frame(self.nav_target, self.world_map1) #目标在原始地图坐标系中位置
            self.map1_start = change_frame(self.world_map2, self.world_map1) #开始点在原始map1中位置
            #判断初始点是否在free space
            start_x = self.map1_start[0]
            start_y = self.map1_start[1]
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
                
            #check whether target in free space 
            nav_target_x = self.map1_target[0]
            nav_target_y = self.map1_target[1]
            target_in_vertex_index = []
            for i in range(len(self.map.vertex)):
                now_free_space = self.map.vertex[i].local_free_space_rect
                if now_free_space[0] < nav_target_x and nav_target_x < now_free_space[2] and now_free_space[1] < nav_target_y and nav_target_y < now_free_space[3]:
                    target_in_vertex_index.append(i)

            if len(target_in_vertex_index) == 0:
                #对于目标点不在free space 情况，找一个最近的vertex导航过去
                min_dis = 1e100
                min_index = -1
                for i in range(len(self.map.vertex)):
                    now_vertex_pose = self.map.vertex[i].pose[0:2]
                    now_dis = ((nav_target_x - now_vertex_pose[0])**2 + (nav_target_y - now_vertex_pose[1])**2)**0.5
                    if now_dis < min_dis:
                        min_dis = now_dis
                        min_index = i
                target_in_vertex_index.append(min_index)

            self.adj_list = dict()
            self.edge_to_adj_list()
            #get total id and pose
            shortest_path_length = 1e100
            target_path = None
            start_point_pose = np.array([start_x,start_y])
            end_point_pose = np.array([nav_target_x,nav_target_y])
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
                    
                    
            # target_id_list = [start_in_vertex_index,target_in_vertex_index]
            # #estimate path on topomap
            # topo_map = topo_map_path(self.adj_list,target_id_list[-1], target_id_list[0:-1])
            # topo_map.get_path()
            # target_path = topo_map.foundPath[0][::-1]

            #这部分最后的输出结果
            self.update_navigation_path_flag = True
            self.target_topo_path = target_path
        else:
            #开始导航
            robot_pose = np.array(self.pose[0:2])
            now_vertex_id = self.target_topo_path[self.now_target_vertex]
            target_vertex_pose = np.array(self.map.vertex[now_vertex_id].pose[0:2])
            target_vertex_pose = change_frame(target_vertex_pose, self.map1_map2) # 换到map2
            target_dis = np.linalg.norm(robot_pose - target_vertex_pose)

            if self.now_target_vertex == -1:#目标是最后一个点
                #判断是否到达最后一个点
                final_target = np.array(self.map2_target[0:2])
                if np.linalg.norm(robot_pose - final_target) < 0.1:#到达最后一个点了
                    print('goal reached!')
                    return
            else:
                map_origin = np.array(self.map_origin)
                target_pose = np.array(self.map2_target[0:2])

                target_vertex_pose_pixel = (target_pose- map_origin)/self.map_resolution
                height, width = self.global_map.shape
                x = int(target_vertex_pose_pixel[0])
                y = int(target_vertex_pose_pixel[1])
                if x > 0 and x < width and y > 0 and y < height and self.global_map[y,x] == 0:
                    self.now_target_vertex = -1#如果终点可见，直接去终点
                    final_target = copy.deepcopy(self.map2_target)
                    final_target[2] = np.rad2deg(final_target[2])
                    goal_message, self.goal = self.get_move_goal(self.self_robot_name,final_target )#offset = 0
                    goal_marker = self.get_goal_marker(self.self_robot_name, final_target)
                    self.actoinclient.send_goal(goal_message)
                    self.goal_pub.publish(goal_marker)
                else:
                    find_new_target = False
                    for i in range(len(self.target_topo_path)-1,self.now_target_vertex,-1):#逆序索引
                        now_point_index = self.target_topo_path[i]
                        target_pose = np.array(self.map.vertex[now_point_index].pose[0:2])
                        target_pose = change_frame(target_pose, self.map1_map2) # 换到map2
                        target_vertex_pose_pixel = (target_pose- map_origin)/self.map_resolution
                        x = int(target_vertex_pose_pixel[0])
                        y = int(target_vertex_pose_pixel[1])
                        if x > 0 and x < width and y > 0 and y < height and self.global_map[y,x] == 0:
                            self.now_target_vertex = i #如果目标可见，直接去目标点
                            find_new_target = True
                            break
                    
                    #检测当前目标周围是否存在障碍物
                    if find_new_target == False:
                        now_target_pose = np.array(self.map.vertex[self.target_topo_path[self.now_target_vertex]].pose[0:2])
                        now_target_pose = change_frame(now_target_pose, self.map1_map2) # 换到map2
                        target_vertex_pose_pixel = (now_target_pose- map_origin)/self.map_resolution
                        x = int(target_vertex_pose_pixel[0])
                        y = int(target_vertex_pose_pixel[1])
                        expaned_width = 5
                        local_map = self.global_map[y-expaned_width:y+expaned_width,x-expaned_width:x+expaned_width]
                        if np.any(local_map==100):
                            if self.now_target_vertex == len(self.target_topo_path) - 1:
                                self.now_target_vertex = -1
                                final_target = copy.deepcopy(self.map2_target)
                                final_target[2] = np.rad2deg(final_target[2])
                                goal_message, self.goal = self.get_move_goal(self.self_robot_name,final_target )#offset = 0
                                goal_marker = self.get_goal_marker(self.self_robot_name, final_target)
                                self.actoinclient.send_goal(goal_message)
                                self.goal_pub.publish(goal_marker)
                                return
                            else:
                                self.now_target_vertex += 1
                            find_new_target = True

                    if not find_new_target:
                        #没有找到新的目标，但是已经到了这一个目标
                        if target_dis < 1:
                            if self.now_target_vertex == len(self.target_topo_path) - 1: #如果到了最后一个点，那么直接去目标点
                                self.now_target_vertex = -1
                                final_target = copy.deepcopy(self.map2_target)
                                final_target[2] = np.rad2deg(final_target[2])
                                goal_message, self.goal = self.get_move_goal(self.self_robot_name,final_target )#offset = 0
                                goal_marker = self.get_goal_marker(self.self_robot_name, final_target)
                                self.actoinclient.send_goal(goal_message)
                                self.goal_pub.publish(goal_marker)
                                return
                            else:
                                self.now_target_vertex += 1
                                find_new_target = True
                        if self.now_target_vertex==0:
                            self.now_target_vertex += 1
                            now_vertex_id = self.target_topo_path[self.now_target_vertex]
                            now_vertex_pose = self.map.vertex[now_vertex_id].pose
                            now_vertex_pose = change_frame(now_vertex_pose, self.map1_map2) # 换到map2
                            now_vertex_pose[2] = np.rad2deg(now_vertex_pose[2])
                            goal_message, self.goal = self.get_move_goal(self.self_robot_name,now_vertex_pose )#offset = 0
                            goal_marker = self.get_goal_marker(self.self_robot_name, now_vertex_pose)
                            self.actoinclient.send_goal(goal_message)
                            self.goal_pub.publish(goal_marker)
                    
                    if find_new_target:#前往新的目标点
                        now_vertex_id = self.target_topo_path[self.now_target_vertex]
                        now_vertex_pose = self.map.vertex[now_vertex_id].pose
                        now_vertex_pose = change_frame(now_vertex_pose, self.map1_map2) # 换到map2
                        now_vertex_pose[2] = np.rad2deg(now_vertex_pose[2])
                        goal_message, self.goal = self.get_move_goal(self.self_robot_name,now_vertex_pose )#offset = 0
                        goal_marker = self.get_goal_marker(self.self_robot_name, now_vertex_pose)
                        self.actoinclient.send_goal(goal_message)
                        self.goal_pub.publish(goal_marker)

            
    def change_goal(self):
        # move goal:now_pos + basic_length+offset;  now_angle + nextmove
        move_goal = self.choose_nav_goal()
        goal_message, self.goal = self.get_move_goal(self.self_robot_name,move_goal )#offset = 0
        goal_marker = self.get_goal_marker(self.self_robot_name, move_goal)
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)

        


    def map_grid_callback(self, data):
        
        if self.vertex_map_ready:
            self.visulize_vertex()
        #generate grid map and global grid map
        range = int(6/self.map_resolution)
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)
        timenow = rospy.Time.now()
        try:
            #robot1/map->robot1/base_footprint
            self.tf_listener.waitForTransform(data.header.frame_id, robot_name+"/base_footprint", timenow, rospy.Duration(0.5))
            tf_transform, rotation = self.tf_listener.lookupTransform(data.header.frame_id, robot_name+"/base_footprint", timenow)
            self.current_loc_pixel = [0,0]
            #data origin position = -13, -12, 0
            self.current_loc_pixel[0] = int((tf_transform[1] - data.info.origin.position.y)/data.info.resolution)
            self.current_loc_pixel[1] = int((tf_transform[0] - data.info.origin.position.x)/data.info.resolution)
            self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
            
            self.global_map_tmp = np.asarray(data.data).reshape(shape)
            self.global_map_tmp[np.where(self.global_map_tmp==-1)] = 255
            self.global_map = copy.deepcopy(self.global_map_tmp)
            #获取当前一个小范围的grid map
            self.grid_map = copy.deepcopy(self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])])
            current_frontier = detect_frontier(self.global_map) * self.map_resolution + np.array(self.map_origin)
            self.total_frontier = np.vstack((self.total_frontier, current_frontier))
            ds_size = 0.1
            tmp, unique_indices = np.unique(np.floor(self.total_frontier / ds_size).astype(int), True, axis=0)
            self.total_frontier = self.total_frontier[unique_indices]

            self.grid_map_ready = 1
            self.update_frontier()
            #保存图片
            if save_result:
                temp = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
                temp[np.where(temp==-1)] = 125
                cv2.imwrite(debug_path+self.self_robot_name + "_local_map.jpg", temp)
                cv2.imwrite(debug_path+self.self_robot_name +"_global_map.jpg", self.global_map)
        except:
            # print("tf listener fails")
            pass
        
        

    def move_base_status_callback(self, data):
        try:
            status = data.status_list[-1].status
        # print(status)
        
            if status >= 3:
                self.erro_count +=1
            if self.erro_count >= 3:
                self.change_goal()
                self.erro_count = 0
                print(self.self_robot_name,"reach error! Using other goal!")
        except:
            pass

    def get_move_goal(self, robot_name, goal)-> MoveBaseGoal():
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()

        orientation = R.from_euler('z', goal[2], degrees=True).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]
        # dont decide which orientation to choose 

        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_message.target_pose.pose.position = pose

        return goal_message, goal

    def get_goal_marker(self, robot_name, goal) -> PoseStamped():
        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        orientation = R.from_euler('z', goal[2], degrees=True).as_quat()
        goal_marker.pose.orientation.x = orientation[0]
        goal_marker.pose.orientation.y = orientation[1]
        goal_marker.pose.orientation.z = orientation[2]
        goal_marker.pose.orientation.w = orientation[3]
        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_marker.pose.position = pose

        return goal_marker

        
    def topomap_callback(self, topomap_message):
        # receive topomap
        if self.receive_topomap:
            return
        self.ready_for_topo_map = False
        Topomap = MessageToTopomap(topomap_message)
        self.map = copy.deepcopy(Topomap)  
        self.received_map = copy.deepcopy(Topomap)  

        self.receive_topomap = True 
        print("-----finish init topomap------")


    
    def free_space_line(self,point1,point2):
        # check whether a line cross a free space
        # point1 and point2 in pixel frame
        now_global_map = self.global_map
        height, width = now_global_map.shape

        # 获取两个点的坐标
        x1, y1 = point1
        x2, y2 = point2

        # 计算两个点之间的距离
        distance = max(abs(x2 - x1), abs(y2 - y1))

        # 计算每个点之间的步长
        step_x = (x2 - x1) / distance
        step_y = (y2 - y1) / distance

        # 检查每个点是否经过的像素都是0
        for i in range(int(distance) + 1):
            x = int(x1 + i * step_x)
            y = int(y1 + i * step_y)
            if x < 0 or x >= width or y < 0 or y >= height or now_global_map[y, x] != 0:
                if now_global_map[y, x] != 255: #unknown
                    return False

        return True

    def is_explored_frontier(self,pose_in_world):
        #input pose in world frame
        expored_range = 1
        frontier_position = np.array([int((pose_in_world[0] - self.map_origin[0])/self.map_resolution), int((pose_in_world[1] - self.map_origin[1])/self.map_resolution)])
        temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
        if np.logical_not(np.any(temp_map == 255)): #unkown place is not in this point
            return True

        expored_range = 4
        temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range, frontier_position[0]-expored_range:frontier_position[0]+expored_range]
        if np.any(np.abs(temp_map - 100) < 40):# delete near obstcal frontier
            return True
        return False

    def update_frontier(self):
        #负责一部分删除未探索方向
        position = self.pose
        position = np.array([position[0], position[1]])
        #delete unexplored direction based on distance between now robot pose and frontier point position
        delete_index = []
        for index, frontier in enumerate(self.total_frontier):
            if self.is_explored_frontier(frontier):
                delete_index.append(index)
        self.total_frontier = np.delete(self.total_frontier, delete_index, axis = 0)

        #goal in map frame
        now_goal = self.goal
        if now_goal.size > 0:
            frontier_position = np.array([int((now_goal[0] - self.map_origin[0])/self.map_resolution), int((now_goal[1] - self.map_origin[1])/self.map_resolution)])
            expored_range = 4
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.any(np.abs(temp_map - 100) < 40):
                # print("Target near obstacle! Change another goal!")
                self.change_goal()
            
            expored_range = 1
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.logical_not(np.any(temp_map == 255)):
                # print("Target is an explored point! Change another goal!")
                self.change_goal()



if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('topo_navigation')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")

    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    
    robot_list.remove(robot_name) #记录其他机器人id
    node = RobotNode(robot_name, robot_list)

    print("-------init robot navigation node--------")
    #订阅自己的图像
    robot1_image1_sub = message_filters.Subscriber(robot_name+"/camera1/image_raw", Image)
    robot1_image2_sub = message_filters.Subscriber(robot_name+"/camera2/image_raw", Image)
    robot1_image3_sub = message_filters.Subscriber(robot_name+"/camera3/image_raw", Image)
    robot1_image4_sub = message_filters.Subscriber(robot_name+"/camera4/image_raw", Image)
    ts = message_filters.TimeSynchronizer([robot1_image1_sub, robot1_image2_sub, robot1_image3_sub, robot1_image4_sub], 10) #传感器信息融合
    ts.registerCallback(node.create_panoramic_callback) # 

    rospy.spin()