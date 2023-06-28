#!/usr/bin/python3.8
from tkinter.constants import Y
import rospy
from rospy.rostime import Duration
from rospy.timer import Rate, sleep
from sensor_msgs.msg import Image, LaserScan
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
from self_topoexplore.msg import ImageWithPointCloudMsg
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
from std_msgs.msg import Header
import sys
from robot_function import *
from RelaPose_2pc_function import *

import subprocess

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

        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        #robot data
        self.pose = [0,0,0] # x y yaw angle
        self.init_map_angle_ready = 0
        self.map_orientation = None
        self.map_angle = None #Yaw angle of map
        self.current_loc_pixel = [0,0]
        self.erro_count = 0
        self.change_another_goal = 0
        self.goal = np.array([10000.0, 10000.0])

        self.laserProjection = LaserProjection()
        self.pcd_queue = Queue(maxsize=10)# no used
        self.detect_loop = 0
        self.grid_map_ready = 0
        self.tf_transform_ready = 0
        self.cv_bridge = CvBridge()
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.last_vertex = -1
        self.current_node = None
        self.last_nextmove = 0 #angle
        self.topomap_meet = 0
        self.vertex_dict = dict()
        self.vertex_dict[self.self_robot_name] = list()
        self.matched_vertex_dict = dict()

        self.edge_dict = dict()
        self.relative_position = dict()
        self.relative_orientation = dict()
        self.meeted_robot = list()
        for item in robot_list:
            self.vertex_dict[item] = list()
            self.matched_vertex_dict[item] = list()
            self.edge_dict[item] = list()
            self.relative_position[item] = [0, 0, 0]
            self.relative_orientation[item] = 0


        self.topomap_matched_score = 0.7
        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        #relative pose estimation
        self.image_req_sub = rospy.Subscriber('/request_image', String, self.receive_image_req_callback)
        self.image_data_sub = None

        self.relative_pose_image = None
        self.relative_pose_pc = None
        self.image_ready = 0
        self.image_req_publisher = rospy.Publisher('/request_image', String, queue_size=1)
        self.image_data_pub = rospy.Publisher(robot_name+'/relative_pose_est_image', ImageWithPointCloudMsg, queue_size=1)#发布自己节点的图片

        x_offset = 0.1
        y_offset = 0.2
        self.cam_trans = [[x_offset,0,0],[0,y_offset,math.pi/2],[-x_offset,0.0,math.pi],[0,-y_offset,-math.pi/2]] # camera position
        self.estimated_vertex_pose = list() #["robot_i","robot_j",id1,id2,estimated_pose] suppose that i < j
        self.map_frame_pose = dict() # map_frame_pose[robot_j] is [R_j,t_j] R_j 3x3
        self.ready_for_topo_map = True
        self.laser_scan_cos_sin = None
        self.laser_scan_init = False
        self.local_laserscan = None
        #move base
        self.actoinclient = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)
        self.trajectory_point = None
        self.trajectory_rate = Rate(0.3)
        self.trajectory_length = 0
        self.vertex_on_path = []
        self.reference_vertex = None
        self.start_time = time.time()

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
        self.frontier_publisher = rospy.Publisher(robot_name+'/frontier_points', Marker, queue_size=1)
            

        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.update_frontier_callback, queue_size=1)#data is not important
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)

        for robot in robot_list:
            rospy.Subscriber(
                robot+"/topomap", TopoMapMsg, self.topomap_callback, queue_size=1, buff_size=52428800)
            rospy.Subscriber(
                robot+"/ud", UnexploredDirectionsMsg, self.unexplored_directions_callback, robot, queue_size=1)
        
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

    def receive_image_callback(self, image_pc):
        image = image_pc.image
        self.relative_pose_image = self.cv_bridge.imgmsg_to_cv2(image)
        self.image_ready = 1

        pc = image_pc.lidar_point
        self.relative_pose_pc = pc

    def receive_image_req_callback(self, req):
        #req: i_j_id
        input_list = req.data.split()  # 将字符串按照空格分割成一个字符串列表
        if input_list[1] in self.self_robot_name:
            # publish image msg
            img_pc_msg = ImageWithPointCloudMsg()
            image_index = int(input_list[2])
            req_image = self.map.vertex[image_index].local_image
            header = Header(stamp=rospy.Time.now())
            image_msg = self.cv_bridge.cv2_to_imgmsg(req_image, encoding="mono8")
            image_msg.header = header
            #use local map for icp
            # pc_image = self.map.vertex[image_index].localMap
            # x, y = np.where((pc_image > 90) & (pc_image < 110))
            # half_image_width = int(pc_image.shape[0]/2)
            # x = x-half_image_width
            # y = y-half_image_width

            #use real laser scan for icp
            x = self.map.vertex[image_index].local_laserscan[0,:]
            y = self.map.vertex[image_index].local_laserscan[1,:]

            pc =  np.concatenate((x, y)).tolist()
            img_pc_msg.image = image_msg
            img_pc_msg.lidar_point = pc
            self.image_data_pub.publish(img_pc_msg) #finish publish message


            print("robot = ",self.self_robot_name,"  publish image index = ",image_index)




    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        #goal pose offset
        offset = 0

        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        self.tf_listener2.waitForTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.1))
        try:
            self.tf_transform, self.rotation = self.tf_listener2.lookupTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow)
            self.tf_transform_ready = 1
            self.pose[0] = self.tf_transform[0]
            self.pose[1] = self.tf_transform[1]
            self.pose[2] = R.from_quat(self.rotation).as_euler('xyz', degrees=True)[2]

            if self.init_map_angle_ready == 0:
                self.map_angle = self.pose[2]
                print("finish create map, map angle = ", self.map_angle)
                self.map.offset_angle = self.map_angle
                self.init_map_angle_ready = 1
        except:
            pass
        current_pose = copy.deepcopy(self.pose)
        # ----finish getting pose----

        #define where to add a new vertex
        pass_cal_feature = False
        if self.last_vertex != -1:
            last_pose = np.array([self.map.vertex[self.last_vertex].pose[0],self.map.vertex[self.last_vertex].pose[1]])
            now_pose = np.array([current_pose[0],self.map.vertex[self.last_vertex].pose[1]])
            dis = np.linalg.norm(last_pose - now_pose)
            #distance between to vertex is near
            if dis < 1.5:
                pass_cal_feature = True
        
        matched_flag = 0
        if not pass_cal_feature:
            #init cvBridge
            panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
            feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
            gray_local_img = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2GRAY)
            vertex = Vertex(robot_name, id=-1, pose=current_pose, descriptor=feature, local_image=gray_local_img, local_laserscan=self.local_laserscan)
            self.last_vertex, self.current_node, matched_flag = self.map.add(vertex, self.last_vertex, self.current_node)
            if matched_flag==0:# add a new vertex
                #create a new one
                while self.grid_map_ready==0 or self.tf_transform_ready==0:
                    time.sleep(0.5)
                # 1. set and publish the topological map visulization markers
                self.vertex_dict[self.self_robot_name].append(vertex.id)

                localMap = self.grid_map
                self.map.vertex[-1].localMap = localMap #every vertex contains local map
                self.detect_loop = 0
                picked_vertex_id = self.map.upgradeFrontierPoints(self.last_vertex,resolution=self.map_resolution)
        
                #  choose navigableDirection when add a new vertex
                print("picked_vertex_id = ", picked_vertex_id)
                navigableDirection = self.map.vertex[picked_vertex_id].navigableDirection
                nextmove = 0
                directionID = 0
                max_dis = 0
                dis_with_other_centers = [0 for i in range(len(self.meeted_robot))]
                dis_scores = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
                dis_with_vertices = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
                epos = 0.2

                if len(navigableDirection) == 0:#no navigable Dirention
                    self.change_another_goal = 1
                else:
                    # choose where to go
                    # 选择前沿点
                    for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints)):
                        position_tmp = self.map.vertex[picked_vertex_id].frontierPoints[i]
                        now_vertex_pose = np.array(self.map.vertex[picked_vertex_id].pose[0:2])
                        # map.center : center of vertex
                        dis_tmp = np.sqrt(np.sum(np.square(position_tmp + now_vertex_pose - self.map.center)))#the point farthest from the center of the map
                        dis_scores[i] += epos * dis_tmp

                    #choose max dis as next move
                    for i in range(len(dis_scores)):
                        if dis_scores[i] > max_dis:
                            max_dis = dis_scores[i]
                            directionID = i
                    move_direction = navigableDirection[directionID]
                    
                    # delete this frontier
                    print("navigableDirection == 0 so delete this frontier")
                    if len(self.map.vertex[picked_vertex_id].navigableDirection)!=0:# choose this navigation direction
                        del self.map.vertex[picked_vertex_id].navigableDirection[directionID]
                        ud_message = UnexploredDirectionsMsg()
                        ud_message.robot_name = self.map.vertex[picked_vertex_id].robot_name
                        ud_message.vertexID = self.map.vertex[picked_vertex_id].id
                        ud_message.directionID = directionID
                        self.unexplore_direction_pub.publish(ud_message)#publish an unexplored direction
                    if len(self.map.vertex[picked_vertex_id].frontierDistance)!=0:
                        basic_length = self.map.vertex[picked_vertex_id].frontierDistance[directionID]
                        del self.map.vertex[picked_vertex_id].frontierDistance[directionID]
                    if len(self.map.vertex[picked_vertex_id].frontierPoints)!=0:
                        del self.map.vertex[picked_vertex_id].frontierPoints[directionID]
                    self.detect_loop = 1#assign as detect loop?
                    
                    #move goal:now_pos + basic_length+offset;  now_angle + nextmove
                    self.last_nextmove = move_direction
                    goal_message, self.goal = self.get_move_goal(robot_name, current_pose, move_direction, basic_length+offset)#offset = 0
                    goal_marker = self.get_goal_marker(robot_name, current_pose, move_direction, basic_length+offset)
                    self.actoinclient.send_goal(goal_message)
                    self.goal_pub.publish(goal_marker)
        
        # ----------visualize frontier------------
        frontier_marker = Marker()
        now = rospy.Time.now()
        frontier_marker.header.frame_id = robot_name + "/map"
        frontier_marker.header.stamp = now
        frontier_marker.ns = "frontier_point"
        frontier_marker.type = Marker.POINTS
        frontier_marker.action = Marker.ADD
        frontier_marker.pose.orientation.w = 1.0
        frontier_marker.scale.x = 0.2
        frontier_marker.scale.y = 0.2
        frontier_marker.color.r = 1.0
        frontier_marker.color.a = 0.5

        for now_vertex in self.map.vertex:
            now_vertex_pose = now_vertex.pose
            for frontier_pos in now_vertex.frontierPoints:
                point_msg = Point()
                point_msg.x = frontier_pos[0] + now_vertex_pose[0]
                point_msg.y = frontier_pos[1] + now_vertex_pose[1]
                point_msg.z = 0.2
                frontier_marker.points.append(point_msg)

        self.frontier_publisher.publish(frontier_marker)
        # --------------finish visualize frontier---------------

        #可视化vertex
        marker_array = MarkerArray()
        marker_message = set_marker(robot_name, len(self.map.vertex), self.map.vertex[0].pose, action=Marker.DELETEALL)
        marker_array.markers.append(marker_message)
        self.marker_pub.publish(marker_array) #DELETEALL 操作，防止重影
        marker_array = MarkerArray()
        markerid = 0
        for vertex in self.map.vertex:
            if vertex.robot_name != robot_name:
                marker_message = set_marker(robot_name, markerid, vertex.pose)#other color
            else:
                marker_message = set_marker(robot_name, markerid, vertex.pose, color=(1,0,0))
            marker_array.markers.append(marker_message)
            markerid += 1
            direction_marker_id = 0
        
        #visualize edge
        #可视化edge就是把两个vertex的pose做一个连线
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
                    edge_message = set_edge(robot_name, edge.id, poses)
                    edge_array.markers.append(edge_message)
                    break
        self.marker_pub.publish(marker_array)
        self.edge_pub.publish(edge_array)

        #deal with no place to go
        if self.change_another_goal:
            print(self.self_robot_name, " no place to go ")
            find_a_goal = 0
            position = np.array([self.pose[0], self.pose[1]])
            distance_list = []
            #计算距离不同节点的距离
            for i in range(len(self.map.vertex)):
                temp_position = self.map.vertex[i].pose
                temp_position = np.asarray([temp_position[0], temp_position[1]])
                distance_list.append(np.linalg.norm(temp_position - position))
            
            while(distance_list):
                min_dis = min(distance_list)
                index = distance_list.index(min_dis)#找到最近的节点
                if len(self.map.vertex[index].navigableDirection) != 0:
                    nextmove = self.map.vertex[index].navigableDirection[0]#直接选择地一个节点进行探索
                    if len(self.map.vertex[index].navigableDirection)!=0:
                        del self.map.vertex[index].navigableDirection[0]
                        ud_message = UnexploredDirectionsMsg()
                        ud_message.robot_name = self.map.vertex[index].robot_name
                        ud_message.vertexID = self.map.vertex[index].id
                        ud_message.directionID = 0
                        self.unexplore_direction_pub.publish(ud_message)
                    if len(self.map.vertex[index].frontierDistance)!=0:
                        basic_length = self.map.vertex[index].frontierDistance[0]
                        del self.map.vertex[index].frontierDistance[0]
                    if len(self.map.vertex[index].frontierPoints)!=0:
                        point = self.map.vertex[index].pose
                        del self.map.vertex[index].frontierPoints[0]
                        print(self.self_robot_name,"delete a frontierPoints")
                    self.detect_loop = 1
                    
                    #move goal:now_pos + basic_length+offset;  now_angle + nextmove
                    nextmove += self.map_angle
                    goal_message, self.goal = self.get_move_goal(robot_name, point, nextmove, basic_length+offset)
                    goal_marker = self.get_goal_marker(robot_name, point, nextmove, basic_length+offset)
                    self.actoinclient.send_goal(goal_message)
                    self.goal_pub.publish(goal_marker)
                    find_a_goal = 1
                    break
                del distance_list[index]
                if find_a_goal:
                    break
            self.change_another_goal = 0



    def map_grid_callback(self, data):
        #generate grid map and global grid map
        range = int(6/self.map_resolution)
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)
        timenow = rospy.Time.now()
        #robot1/map->robot1/base_footprint
        self.tf_listener.waitForTransform(data.header.frame_id, robot_name+"/base_footprint", timenow, rospy.Duration(0.5))
        try:
            tf_transform, rotation = self.tf_listener.lookupTransform(data.header.frame_id, robot_name+"/base_footprint", timenow)
            self.current_loc_pixel = [0,0]
            #data origin position = -13, -12, 0
            self.current_loc_pixel[0] = int((tf_transform[1] - data.info.origin.position.y)/data.info.resolution)
            self.current_loc_pixel[1] = int((tf_transform[0] - data.info.origin.position.x)/data.info.resolution)
            self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
            
            self.global_map = np.asarray(data.data).reshape(shape)
            #获取当前一个小范围的grid map
            self.grid_map = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
            self.grid_map[np.where(self.grid_map==-1)] = 255
            self.grid_map_ready = 1
            self.global_map[np.where(self.global_map==-1)] = 255
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
                self.change_another_goal = 1
                self.erro_count = 0
                print(self.self_robot_name,"reach error! Using other goal!")
        except:
            pass

    def get_move_goal(self, robot_name, now_robot_pose, move_direction, move_length=4)-> MoveBaseGoal():
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()

        orientation = R.from_euler('z', move_direction, degrees=True).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]

        pose = Point()
        move_direction = math.radians(move_direction)
        x = now_robot_pose[0] + move_length * np.cos(move_direction)
        y = now_robot_pose[1] + move_length * np.sin(move_direction)
        goal = np.array([x, y])
        pose.x = x
        pose.y = y
        goal_message.target_pose.pose.position = pose

        return goal_message, goal

    def get_goal_marker(self, robot_name, now_robot_pose, move_direction, move_length=4) -> PoseStamped():
        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        
        orientation = R.from_euler('z', move_direction, degrees=True).as_quat()
        goal_marker.pose.orientation.x = orientation[0]
        goal_marker.pose.orientation.y = orientation[1]
        goal_marker.pose.orientation.z = orientation[2]
        goal_marker.pose.orientation.w = orientation[3]

        pose = Point()
        move_direction = math.radians(move_direction)
        pose.x = now_robot_pose[0] + move_length * np.cos(move_direction)
        pose.y = now_robot_pose[1] + move_length * np.sin(move_direction)

        goal_marker.pose.position = pose

        return goal_marker

    def is_explored_frontier(self,pose_in_world):
        #input pose in world frame
        expored_range = 20
        location = [0, 0]
        frontier_position = np.array([int((pose_in_world[0] - self.map_origin[0])/self.map_resolution), int((pose_in_world[1] - self.map_origin[1])/self.map_resolution)])
        shape = self.global_map.shape
        temp_map = self.global_map[max(frontier_position[0]-expored_range,0):min(frontier_position[0]+expored_range,shape[0]), max(location[1]-expored_range,0):min(location[1]+expored_range, shape[1])]
               
        result = np.logical_not(np.any(temp_map == 255)) #unkown place is not in this point
        return result
            
    def update_frontier_callback(self, data):
        #负责一部分删除未探索方向
        position = self.pose
        position = np.array([position[0], position[1]])
        # if self.detect_loop:
        #delete unexplored direction based on distance between now robot pose and frontier point position
        for i in range(len(self.map.vertex)):
            meeted = list()
            vertex = self.map.vertex[i]
            now_vertex_pose = np.array([vertex.pose[0],vertex.pose[1]])

            for j in range(len(vertex.navigableDirection)-1, -1, -1):
                unexplored = vertex.frontierPoints[j] + now_vertex_pose # change into map frame

                if np.sqrt(np.sum(np.square(position-unexplored))) < 3:#delete unexplored position
                    if self.is_explored_frontier(unexplored):
                        meeted.append(j)
            for index in meeted:
                del self.map.vertex[i].navigableDirection[index]
                del self.map.vertex[i].frontierDistance[index]
                del self.map.vertex[i].frontierPoints[index]
                ud_message = UnexploredDirectionsMsg()
                ud_message.robot_name = vertex.robot_name
                ud_message.vertexID = vertex.id
                ud_message.directionID = index
                self.unexplore_direction_pub.publish(ud_message)

        #delete unexplored direction based on direction
        # if self.reference_vertex == None:
        #     self.reference_vertex = self.current_node
        # rr = int(6/self.map_resolution)
        # for i in range(len(self.map.vertex)):
        #     vertex = self.map.vertex[i]
        #     vertex_position = np.array([vertex.pose[0], vertex.pose[1]])
        #     if vertex.navigableDirection:
        #         if np.sqrt(np.sum(np.square(position-vertex_position))) < 100:
        #             location = [0, 0]
        #             shape = self.global_map.shape
        #             location[0] = self.current_loc_pixel[0] + int((self.pose[0] - vertex_position[0])/self.map_resolution)
        #             location[1] = self.current_loc_pixel[1] - int((self.pose[1] - vertex_position[1])/self.map_resolution)
        #             temp_map = self.global_map[max(location[0]-rr,0):min(location[0]+rr,shape[0]), max(location[1]-rr,0):min(location[1]+rr, shape[1])]
        #             self.map.vertex[i].localMap = temp_map #renew local map
        #             old_ud = self.map.vertex[i].navigableDirection
        #             new_ud = self.map.vertex[i].navigableDirection
        #             delete_list = []
        #             for j in range(len(old_ud)):
        #                 not_deleted = 1
        #                 for uds in new_ud:
        #                     if abs(old_ud[j]-uds) < 5:
        #                         not_deleted = 0
        #                 if not_deleted == 1:
        #                     delete_list.append(j)
        #             for uds in delete_list:
        #                 ud_message = UnexploredDirectionsMsg()
        #                 ud_message.robot_name = self.map.vertex[i].robot_name
        #                 ud_message.vertexID = self.map.vertex[i].id
        #                 ud_message.directionID = uds
        #                 self.unexplore_direction_pub.publish(ud_message)

        #     if np.linalg.norm(position-vertex_position) < 2.5:#现在机器人位置距离节点位置很近
        #         new_vertex_on_path = 1
        #         for svertex in self.vertex_on_path:
        #             if svertex.robot_name == vertex.robot_name and svertex.id == vertex.id:
        #                 new_vertex_on_path = 0
        #         if new_vertex_on_path == 1 and vertex.robot_name!= self.reference_vertex.robot_name and vertex.id != self.reference_vertex.id:
        #             self.vertex_on_path.append(vertex)
        #     if np.linalg.norm(self.goal - vertex_position) < 3:
        #         self.vertex_on_path.append(vertex)

        if len(self.map.vertex) != 0:
            topomap_message = TopomapToMessage(self.map)
            self.topomap_pub.publish(topomap_message) # publish topomap important!
        
        # if len(self.vertex_on_path) >= 3:
        #     self.change_another_goal = 1
        #     self.vertex_on_path = []
        #     self.reference_vertex = self.current_node

    def topomap_callback(self, topomap_message):
        # receive topomap from other robots
        if not self.ready_for_topo_map:
            return
        update_robot_center = False
        self.ready_for_topo_map = False
        Topomap = MessageToTopomap(topomap_message)
        matched_vertex = list()

        # find max matched vertex
        for vertex in Topomap.vertex:
            if vertex.robot_name==self.self_robot_name or vertex.id in self.matched_vertex_dict[vertex.robot_name]:
                # already added vertex or vertex belong to this robot
                pass
            else:
                max_score = 0
                max_index = -1
                for index2, svertex in enumerate(self.map.vertex):#match vertex
                    if svertex.robot_name == vertex.robot_name:#created by same robot
                            pass
                    else:
                        score = np.dot(vertex.descriptor.T, svertex.descriptor)
                        if score > self.topomap_matched_score and score > max_score:
                            max_index = index2
                            max_score = score

                # if matched calculated relative pose
                if max_score > 0:
                    now_matched_vertex = self.map.vertex[max_index]
                    print("matched: now robot = ", now_matched_vertex.robot_name, now_matched_vertex.id,"; target robot = ", vertex.robot_name, vertex.id, score)
                    if vertex.robot_name not in self.meeted_robot:
                        self.meeted_robot.append(vertex.robot_name)
                    self.matched_vertex_dict[vertex.robot_name].append(vertex.id)
                    #estimate relative position
                    #请求一下图片
                    self.image_data_sub  = rospy.Subscriber(vertex.robot_name+"/relative_pose_est_image", ImageWithPointCloudMsg, self.receive_image_callback)
                    req_string = self.self_robot_name[-1] +" "+vertex.robot_name[-1]+" "+str(vertex.id)
                    while not self.image_ready:
                        # 发布请求图片的消息
                        self.image_req_publisher.publish(req_string)
                        rospy.sleep(0.1)
                    #unsubscrib image topic 
                    self.image_data_sub.unregister()
                    self.image_data_sub = None
                    self.image_ready = 0

                    img1 = now_matched_vertex.local_image
                    img2 = self.relative_pose_image #from other robot

                    #use local map
                    # pc1_image = svertex.localMap
                    # half_img_width = int(pc1_image.shape[0]/2)

                    # x1, y1 = np.where((pc1_image > 90) & (pc1_image < 110))
                    # x2_y2 = np.array(self.relative_pose_pc).reshape((2,-1))

                    # pc1 = np.vstack((x1 - half_img_width, y1 - half_img_width, np.zeros(x1.shape,dtype=float))) * self.map_resolution
                    # pc2 = np.vstack((x2_y2, np.zeros(x2_y2.shape[1],dtype=float))) * self.map_resolution
                    
                    #use local laser scan
                    x2_y2 = np.array(self.relative_pose_pc).reshape((2,-1))
                    pc1 = np.vstack((now_matched_vertex.local_laserscan, np.zeros(now_matched_vertex.local_laserscan.shape[1],dtype=float)))
                    pc2 = np.vstack((x2_y2, np.zeros(x2_y2.shape[1],dtype=float)))
                    #save result
                    if save_result:
                        tmp = (pc1, pc2)
                        cv2.imwrite(debug_path+self.self_robot_name + "_self"+str(now_matched_vertex.id)+".jpg", img1)
                        cv2.imwrite(debug_path+self.self_robot_name + "_received"+str(now_matched_vertex.id)+".jpg", img2)
                        np.savez(debug_path + self.self_robot_name + 'pc_data'+ str(now_matched_vertex.id)+'.npz', *tmp)
                    #estimated pose
                    pose = planar_motion_calcu_mulit(img1,img2,k1 = self.K_mat,k2 = self.K_mat,cam_pose = self.cam_trans, pc1=pc1,pc2=pc2)
                    print("estimated pose is:\n",pose)
                    if pose is None:
                        continue
                    print("vertex id and pose:\n",vertex.id,"  ",vertex.pose)
                    self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,now_matched_vertex.pose,list(vertex.pose),pose])
                    
                    matched_vertex.append(vertex)

                    #add edge between meeted two robot
                    tmp_link = [[now_matched_vertex.robot_name, now_matched_vertex.id], [vertex.robot_name, vertex.id]]
                    self.map.edge.append(Edge(id=self.map.edge_id, link=tmp_link))
                    self.map.edge_id += 1
                    #estimate the center of other robot     
                    update_robot_center = True


        # for vertex in Topomap.vertex:
        #     if vertex.robot_name==self.self_robot_name or vertex.id in self.matched_vertex_dict[vertex.robot_name]:
        #         # already added vertex or vertex belong to this robot
        #         pass
        #     else:
        #         # start match
        #         for svertex in self.map.vertex:#match vertex
        #             if svertex.robot_name == vertex.robot_name:#created by same robot
        #                 pass
        #             else:
        #                 score = np.dot(vertex.descriptor.T, svertex.descriptor)
        #                 if score > self.topomap_matched_score:
        #                     print("matched: now robot = ", svertex.robot_name, svertex.id,"; target robot = ", vertex.robot_name, vertex.id, score)
        #                     if vertex.robot_name not in self.meeted_robot:
        #                         self.meeted_robot.append(vertex.robot_name)
        #                     self.matched_vertex_dict[vertex.robot_name].append(vertex.id)
        #                     #estimate relative position
        #                     #请求一下图片
        #                     self.image_data_sub  = rospy.Subscriber(vertex.robot_name+"/relative_pose_est_image", ImageWithPointCloudMsg, self.receive_image_callback)
        #                     req_string = self.self_robot_name[-1] +" "+vertex.robot_name[-1]+" "+str(vertex.id)
        #                     while not self.image_ready:
        #                         # 发布请求图片的消息
        #                         self.image_req_publisher.publish(req_string)
        #                         rospy.sleep(0.1)
        #                     #unsubscrib image topic 
        #                     self.image_data_sub.unregister()
        #                     self.image_data_sub = None

        #                     img1 = svertex.local_image
        #                     img2 = self.relative_pose_image #from other robot

        #                     #use local map
        #                     # pc1_image = svertex.localMap
        #                     # half_img_width = int(pc1_image.shape[0]/2)

        #                     # x1, y1 = np.where((pc1_image > 90) & (pc1_image < 110))
        #                     # x2_y2 = np.array(self.relative_pose_pc).reshape((2,-1))

        #                     # pc1 = np.vstack((x1 - half_img_width, y1 - half_img_width, np.zeros(x1.shape,dtype=float))) * self.map_resolution
        #                     # pc2 = np.vstack((x2_y2, np.zeros(x2_y2.shape[1],dtype=float))) * self.map_resolution
                            
        #                     #use local laser scan
        #                     x2_y2 = np.array(self.relative_pose_pc).reshape((2,-1))
        #                     pc1 = np.vstack((svertex.local_laserscan, np.zeros(svertex.local_laserscan.shape[1],dtype=float)))
        #                     pc2 = np.vstack((x2_y2, np.zeros(x2_y2.shape[1],dtype=float)))
        #                     #save result
        #                     tmp = (pc1, pc2)
        #                     cv2.imwrite(debug_path+self.self_robot_name + "_self"+str(svertex.id)+".jpg", img1)
        #                     cv2.imwrite(debug_path+self.self_robot_name + "_received"+str(svertex.id)+".jpg", img2)
        #                     np.savez(debug_path + self.self_robot_name + 'pc_data'+ str(svertex.id)+'.npz', *tmp)
        #                     #estimated pose
        #                     pose = planar_motion_calcu_mulit(img1,img2,k1 = self.K_mat,k2 = self.K_mat,cam_pose = self.cam_trans, pc1=pc1,pc2=pc2)
        #                     print("estimated pose is:\n",pose)
        #                     if pose is None:
        #                         continue
        #                     print("vertex id and pose:\n",vertex.id,"  ",vertex.pose)
        #                     self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.pose,list(vertex.pose),pose])
                            
        #                     matched_vertex.append(vertex)

        #                     #add edge between meeted two robot
        #                     tmp_link = [[svertex.robot_name, svertex.id], [vertex.robot_name, vertex.id]]
        #                     self.map.edge.append(Edge(id=self.map.edge_id, link=tmp_link))
        #                     self.map.edge_id += 1
        #                     #estimate the center of other robot     
        #                     update_robot_center = True
        #                     break
        
        if update_robot_center:
            self.topo_optimize()
            for edge in Topomap.edge:
                if edge.link[0][1] in self.vertex_dict[edge.link[0][0]]:
                    if edge.link[1][1] in self.vertex_dict[edge.link[1][0]] or edge.link[1][0]==self.self_robot_name:
                        pass
                    else:
                        edge.id = self.map.edge_id
                        self.map.edge_id += 1
                        self.map.edge.append(edge)
                elif edge.link[1][1] in self.vertex_dict[edge.link[1][0]]:
                    if edge.link[0][1] in self.vertex_dict[edge.link[0][0]] or edge.link[0][0]==self.self_robot_name:
                        pass
                    else:
                        edge.id = self.map.edge_id
                        self.map.edge_id += 1
                        self.map.edge.append(edge)
                else:
                    edge.id = self.map.edge_id
                    self.map.edge_id += 1
                    self.map.edge.append(edge)
            for vertex in Topomap.vertex:
                #add vertex
                if vertex.id not in self.vertex_dict[vertex.robot_name] and vertex.robot_name != self.self_robot_name:
                    now_robot_map_frame_rot = self.map_frame_pose[vertex.robot_name][0]
                    now_robot_map_frame_trans = self.map_frame_pose[vertex.robot_name][1]
                    tmp_pose = now_robot_map_frame_rot @ np.array([vertex.pose[0],vertex.pose[1],0]) + now_robot_map_frame_trans
                    vertex.pose = list(vertex.pose)
                    vertex.pose[0] = tmp_pose[0]
                    vertex.pose[1] = tmp_pose[1]
                    self.map.vertex.append(vertex)
                    self.map.vertex_id +=1
                    self.vertex_dict[vertex.robot_name].append(vertex.id)
        
        self.ready_for_topo_map = True
        



    def unexplored_directions_callback(self, data, rn):
        #delete this direction
        for i in range(len(self.map.vertex)):
            vertex = self.map.vertex[i]
            if vertex.robot_name == data.robot_name:
                if vertex.id == data.vertexID:
                    try:
                        del self.map.vertex[i].navigableDirection[data.directionID]
                    except:
                        pass

    def trajectory_length_callback(self, data):
        if self.trajectory_point == None:
            self.trajectory_point = data.markers[2].points[-1]
        temp_position = data.markers[2].points[-1]
        point1 = np.asarray([self.trajectory_point.x, self.trajectory_point.y])
        point2 = np.asarray([temp_position.x, temp_position.y])
        self.trajectory_length += np.linalg.norm(point1 - point2)
        self.trajectory_point = temp_position
        # print(robot_name, "length", self.trajectory_length)

    def topo_optimize(self):
        #self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.id,vertex.id,pose])
        # This part should update self.map_frame_pose[vertex.robot_name];self.map_frame_pose[vertex.robot_name][0] R33;[1]t 31
        input = self.estimated_vertex_pose
        print(input)
        now_meeted_robot_num = len(self.meeted_robot)
        name_list = [self.self_robot_name] + self.meeted_robot
        c_real = [[0,0,0] for i in range(now_meeted_robot_num + 1)]

        now_id = 1
        trans_data = ""
        for center in c_real:
            trans_data+="VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n".format(now_id,center[0],center[1],center[2]/180*math.pi)
            now_id +=1

        for now_input in input:
            # pose_origin1 = numpy.append(pose_origin1, numpy.array([[now_input[2][0]],[now_input[2][1]]]), axis=1)
            # pose_origin2 = numpy.append(pose_origin2, numpy.array([[now_input[3][0]],[now_input[3][1]]]), axis=1)
            now_trust = 1
            start_idx = str(name_list.index(now_input[0])+1)
            end_idx = str(name_list.index(now_input[1])+1)
            trans_data+="EDGE_SE2 {} {} ".format(start_idx,end_idx)
            for j in range(3):
                for k in range(3):
                    trans_data += " {:.6f} ".format(now_input[2+j][k])
            trans_data += " {:.6f} 0 0 {:.6f} 0 {:.6f}\n".format(now_trust,now_trust,now_trust)
            now_id += 2

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建可执行文件的相对路径
        executable_path = os.path.join(current_dir, '..', 'src', 'pose_graph_opt', 'pose_graph_2d')
        process = subprocess.Popen(executable_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        # 向C++程序输入数据
        process.stdin.write(trans_data)
        # 关闭输入流
        process.stdin.close()
        output_data = process.stdout.read()
        # 等待C++程序退出
        process.wait()

        output_data = output_data[:-1]

        rows = output_data.split('\n')
        # 将每行分割成字符串数组
        data_list = [row.split() for row in rows]
        # 将字符串数组转换为浮点数数组
        data_arr = np.array(data_list, dtype=float)
        poses_optimized = data_arr[:,1:]
        poses_optimized[:,-1] = poses_optimized[:,-1] / math.pi *180#转换到角度制度
        # print("estimated pose is:\n", poses_optimized)
        # if self.self_robot_name == "robot1":
        #     poses_optimized = np.array([[0,0,0],[7,7,0]])
        # else:
        #     poses_optimized = np.array([[0,0,0],[-7,-7,0]])
        for i in range(0,now_meeted_robot_num):
            now_meeted_robot_pose = poses_optimized[1+i,:]
            print("---------------Robot Center Optimized-----------------------\n")
            print(self.self_robot_name,"estimated robot pose of ", self.meeted_robot[i],now_meeted_robot_pose)
            self.map_frame_pose[self.meeted_robot[i]] = list()
            self.map_frame_pose[self.meeted_robot[i]].append(R.from_euler('z', now_meeted_robot_pose[2], degrees=True).as_matrix()) 
            self.map_frame_pose[self.meeted_robot[i]].append(np.array([now_meeted_robot_pose[0],now_meeted_robot_pose[1],0]))


if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('topological_map')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    print(robot_name, robot_num)

    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    
    robot_list.remove(robot_name) #记录其他机器人id
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