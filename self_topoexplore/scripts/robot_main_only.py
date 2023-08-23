#!/usr/bin/python3.8
#仅有主节点的方法，消融实验
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
from visualization_msgs.msg import Marker, MarkerArray
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
from TopoMap import Support_Vertex, Vertex, Edge, TopologicalMap
from utils.imageretrieval.imageretrievalnet import init_network
from utils.imageretrieval.extract_feature import cal_feature
from utils.topomap_bridge import TopomapToMessage, MessageToTopomap
from utils.astar import grid_path, topo_map_path
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
from robot_function import *
from RelaPose_2pc_function import *

import subprocess
import scipy.ndimage
import signal


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
        #color: frontier, main_vertex, support vertex, edge
        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        
        #robot data
        self.pose = [0,0,0] # x y yaw angle in degree
        self.init_map_angle_ready = 0
        self.map_orientation = None
        self.map_angle = None #Yaw angle of map
        self.current_loc_pixel = [0,0]
        self.erro_count = 0
        self.goal = np.array([])

        self.laserProjection = LaserProjection()
        self.pcd_queue = Queue(maxsize=10)# no used
        self.grid_map_ready = 0
        self.tf_transform_ready = 0
        self.cv_bridge = CvBridge()
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.last_free_vertex = None #last free support vertex
        self.last_vertex_id = -1
        self.current_node = None
        self.last_nextmove = 0 #angle
        self.topomap_meet = 0
        self.vertex_map_ready = False
        self.vertex_dict = dict()
        self.vertex_dict[self.self_robot_name] = list()
        self.matched_vertex_dict = dict()
        self.now_feature = np.array([])
        self.adj_list = dict()

        self.edge_dict = dict()
        self.relative_position = dict()
        self.relative_orientation = dict()
        self.meeted_robot = list()
        self.potential_main_vertex = list()
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
        self.local_laserscan_angle = None
        #move base
        self.actoinclient = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)
        self.trajectory_point = None
        self.trajectory_rate = Rate(0.3)
        self.trajectory_length = 0
        self.vertex_on_path = []
        self.reference_vertex = None
        self.start_time = time.time()
        self.total_frontier = np.array([],dtype=float).reshape(-1,2)
        self.first_feature = None
        self.finish_explore = False

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
        self.start_pub = rospy.Publisher(
            "/start_exp", String, queue_size=1) #发一个start
        self.frontier_publisher = rospy.Publisher(robot_name+'/frontier_points', Marker, queue_size=1)
        self.vertex_free_space_pub = rospy.Publisher(robot_name+'/vertex_free_space', MarkerArray, queue_size=1)
        self.find_better_path_pub = rospy.Publisher(robot_name+'/find_better_path', String, queue_size=100)
            
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)
        for robot in robot_list:
            rospy.Subscriber(
                robot+"/topomap", TopoMapMsg, self.topomap_callback, queue_size=1, buff_size=52428800)
        rospy.Subscriber(
            robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/scan", LaserScan, self.laserscan_callback, queue_size=1)
        rospy.Subscriber(robot_name+'/find_better_path', String, self.find_better_path_callback, queue_size=100)

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
        self.local_laserscan  = np.array(ranges[valid_indices] * self.laser_scan_cos_sin[:,valid_indices])
        self.local_laserscan_angle = ranges


    def create_panoramic_callback(self, image1, image2, image3, image4):
        #合成一张全景图片然后发布
        img1 = self.cv_bridge.imgmsg_to_cv2(image1, desired_encoding="rgb8")
        img2 = self.cv_bridge.imgmsg_to_cv2(image2, desired_encoding="rgb8")
        img3 = self.cv_bridge.imgmsg_to_cv2(image3, desired_encoding="rgb8")
        img4 = self.cv_bridge.imgmsg_to_cv2(image4, desired_encoding="rgb8")
        panoram = [img1, img2, img3, img4]
        self.panoramic_view = np.hstack(panoram)
        if save_result:
            cv.imwrite(debug_path + "/2.png",self.panoramic_view)
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

            #use real laser scan for icp
            local_laserscan_angle = self.map.vertex[image_index].local_laserscan_angle
            valid_indices = np.isfinite(local_laserscan_angle)
            local_laserscan  = np.array(local_laserscan_angle * self.laser_scan_cos_sin)[:,valid_indices]
            x = local_laserscan[0,:]
            y = local_laserscan[1,:]

            pc =  np.concatenate((x, y)).tolist()
            img_pc_msg.image = image_msg
            img_pc_msg.lidar_point = pc
            self.image_data_pub.publish(img_pc_msg) #finish publish message


            print("robot = ",self.self_robot_name,"  publish image index = ",image_index)


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
            marker.header.frame_id = robot_name + "/map"
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
        marker_array = MarkerArray()
        marker_message = set_marker(robot_name, len(self.map.vertex), self.map.vertex[0].pose, action=Marker.DELETEALL)
        marker_array.markers.append(marker_message)
        self.marker_pub.publish(marker_array) #DELETEALL 操作，防止重影
        marker_array = MarkerArray()
        markerid = 0
        main_vertex_color = (self.vis_color[1][0], self.vis_color[1][1], self.vis_color[1][2])
        support_vertex_color = (self.vis_color[2][0], self.vis_color[2][1], self.vis_color[2][2])

        for vertex in self.map.vertex:
            if vertex.robot_name != robot_name:
                marker_message = set_marker(robot_name, markerid, vertex.pose)#other color
            else:
                if isinstance(vertex, Vertex):
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=main_vertex_color, scale=0.3)
                else:
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=support_vertex_color, scale=0.25)
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
                    edge_message = set_edge(robot_name, edge.id, poses, "edge",main_edge_color, scale=0.1)
                    edge_array.markers.append(edge_message)
                    break
        self.marker_pub.publish(marker_array)
        self.edge_pub.publish(edge_array)


    def choose_exp_goal(self):
        # choose next goal
        dis_epos = 1
        angle_epos = 2
        
        if len(self.total_frontier) == 0:
            return
        total_fontier = copy.deepcopy(self.total_frontier)
        local_range = 10
        #global planner
        # local_index = np.where(np.sqrt(np.sum(np.square(total_fontier - self.pose[0:2]), axis=1)) < local_range)

        #local planner
        # frontier_poses = total_fontier[local_index]
        frontier_poses = total_fontier
        dis_frontier_poses = np.sqrt(np.sum(np.square(frontier_poses - self.pose[0:2]), axis=1))
        dis_tmp = np.exp(-(dis_frontier_poses - 5.6)**2 / 8)

        angle_frontier_poses = np.arctan2(frontier_poses[:, 1] - self.pose[1], frontier_poses[:, 0] - self.pose[0]) - self.pose[2] / 180 * np.pi
        angle_frontier_poses = np.arctan2(np.sin(angle_frontier_poses), np.cos(angle_frontier_poses)) # turn to -pi~pi
        angle_tmp = np.exp(-angle_frontier_poses**2 / 1)
        # calculate frontier information
        vertex_info = np.array(calculate_vertex_info(frontier_poses))

        frontier_scores = (dis_epos * dis_tmp + angle_epos * angle_tmp) / (1 + np.exp(-vertex_info))
        max_index = np.argmax(frontier_scores)


        return frontier_poses[max_index]


    def create_a_vertex(self,panoramic_view):
        #return 1 for uncertainty value reach th; 2 for not a free space line
        #and 0 for don't creat a vertex

        #check whether create a main vertex
        uncertainty_value = 0
        
        main_vertex_dens = 4 #main_vertex_dens^0.5 is the average distance of a vertex, 4 is good
        global_vertex_dens = 2 # create a support vertex large than 2 meter
        now_pose = np.array(self.pose[0:2])
        for now_vertex in self.map.vertex:
            if isinstance(now_vertex, Support_Vertex):
                continue
            now_vertex_pose = np.array(now_vertex.pose[0:2])
            dis = np.linalg.norm(now_vertex_pose - now_pose)
            # print(now_vertex.descriptor_infor)
            uncertainty_value += now_vertex.descriptor_infor * np.exp(-dis**2 / main_vertex_dens)
        if uncertainty_value > 0.57:
            self.potential_main_vertex = list()
        else:
            self.now_feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
            gray_local_img = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2GRAY)
            vertex = Vertex(robot_name, id=-1, pose=copy.deepcopy(self.pose), descriptor=copy.deepcopy(self.now_feature), local_image=gray_local_img, local_laserscan_angle=copy.deepcopy(self.local_laserscan_angle))
            self.potential_main_vertex.append(vertex) 
            if uncertainty_value <0.368:
                #evaluate vertex information
                return 1

        #check wheter create a supportive vertex
        map_origin = np.array(self.map_origin)
        now_robot_pose = (now_pose - map_origin)/self.map_resolution
        free_line_flag = False
        
        for last_vertex in self.map.vertex:
            last_vertex_pose = np.array(last_vertex.pose[0:2])
            last_vertex_pose_pixel = ( last_vertex_pose- map_origin)/self.map_resolution
            if isinstance(last_vertex, Support_Vertex):
                # free_line_flag = self.free_space_line(last_vertex_pose_pixel, now_robot_pose)
                free_line_flag = self.expanded_free_space_line(last_vertex_pose_pixel, now_robot_pose, 1)
            else:   
                free_line_flag = self.expanded_free_space_line(last_vertex_pose_pixel, now_robot_pose, 3)
            
            if free_line_flag:
                break
        
        if not free_line_flag:#if not a line in free space, create a support vertex
            self.potential_main_vertex = list()
            self.now_feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
            gray_local_img = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2GRAY)
            vertex = Vertex(robot_name, id=-1, pose=self.pose, descriptor=self.now_feature, local_image=gray_local_img, local_laserscan_angle=self.local_laserscan_angle)
            self.potential_main_vertex.append(vertex) 
            return 1

        # min_dens_flag = False
        # for last_vertex in self.map.vertex:
        #     last_vertex_pose = np.array(last_vertex.pose[0:2])
        #     if np.linalg.norm(now_pose - last_vertex_pose) < global_vertex_dens:
        #         min_dens_flag = True

        # if not min_dens_flag:#if robot in a place with not that much vertex, then create a support vertex
        #     return 3
        
        return 0


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
    

    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        self.update_robot_pose() #update robot pose
        
        current_pose = copy.deepcopy(self.pose)
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        
        create_a_vertex_flag = self.create_a_vertex(panoramic_view)
        #check whether the robot stop moving
        if not self.finish_explore and len(self.total_frontier) == 0: 
            print("----------Robot Exploration Finished!-----------")
            self.map.vertex[-1].local_free_space_rect  = find_local_max_rect(self.global_map, self.map.vertex[-1].pose[0:2], self.map_origin, self.map_resolution)
            self.visulize_vertex()
            process = subprocess.Popen( "rosbag record -o /home/master/topomap.bag /robot1/topomap /robot1/map", shell=True) #change to your file path
            time.sleep(5)
            # 发送SIGINT信号给进程，让它结束记录
            os.kill(process.pid, signal.SIGINT)
            print("----------FHT-Map Record Finished!-----------")
            print("----------You can use this map for navigation now!-----------")
            self.finish_explore = True

        if create_a_vertex_flag: # create a vertex
            if create_a_vertex_flag == 1:#create a main vertex
                max_infor_index = 0
                max_infor = 0
                for index, now_vertex in enumerate(self.potential_main_vertex):
                    if now_vertex.descriptor_infor > max_infor:
                        max_infor_index = index
                        max_infor = now_vertex.descriptor_infor
                vertex = copy.deepcopy(self.potential_main_vertex[max_infor_index])
                self.last_vertex_id, self.current_node = self.map.add(vertex)
                
            elif create_a_vertex_flag == 2 or create_a_vertex_flag == 3:#create a support vertex
                vertex = Support_Vertex(robot_name, id=-1, pose=current_pose)
                self.last_vertex_id, self.current_node = self.map.add(vertex)
            # add rect to vertex
            if self.last_vertex_id > 0:
                self.map.vertex[-2].local_free_space_rect  = find_local_max_rect(self.global_map, self.map.vertex[-2].pose[0:2], self.map_origin, self.map_resolution)
            #create edge
            self.create_edge()

            self.vertex_map_ready = True
            while self.grid_map_ready==0 or self.tf_transform_ready==0:
                time.sleep(0.5)
            self.vertex_dict[self.self_robot_name].append(vertex.id)
            self.change_goal()
            
            #禁用优化部分
            # if create_a_vertex_flag ==1 or create_a_vertex_flag ==2: 
            #     refine_topo_map_msg = String()
            #     refine_topo_map_msg.data = "Start_find_path!"
            #     self.find_better_path_pub.publish(refine_topo_map_msg) #find a better path
        
    
    def find_better_path_callback(self,data):
        #start compare distance between grid map and topo map
        #estimate distance in topo map
        if len(self.map.edge) == 0:
            return
        
        self.adj_list = dict()
        self.edge_to_adj_list()
        #get total id and pose
        target_id_list = []
        target_pose_list = []
        map_origin = np.array(self.map_origin)
        now_global_map = copy.deepcopy(self.global_map)
        now_global_map_expand = expand_obstacles(now_global_map, 2) 
        for now_vertex in self.map.vertex:
            #turn pose into map frame
            vertex_pose = (np.array(now_vertex.pose[0:2]) - map_origin)/self.map_resolution
            if now_global_map_expand[int(vertex_pose[1]), int(vertex_pose[0])] == 0:
                #make sure this vertex is free on grid map
                target_pose_list.append((int(vertex_pose[1]), int(vertex_pose[0]))) #(y,x) format
                target_id_list.append(now_vertex.id)
        if len(target_pose_list) < 2:
            # print("every vertex is on expanded grid map!")
            return
        #estimate path on topomap
        topo_map = topo_map_path(self.adj_list,target_id_list[-1], target_id_list[0:-1])
        topo_map.get_path()
        topopath_length = np.array(topo_map.path_length)
        topo_path_vertex_num = np.array([len(now_path) for now_path in topo_map.foundPath])
        #estimate path on grid map
        distance_map = scipy.ndimage.distance_transform_edt(now_global_map == 0)
        calculate_grid_path = grid_path(now_global_map_expand,distance_map,target_pose_list[-1], target_pose_list[0:-1])
        calculate_grid_path.get_path()
        grid_path_length = np.array(calculate_grid_path.path_length) * self.map_resolution #(y,x) format

        
        if len(grid_path_length) == 0:
            print("-----------Failed to find a path in grid map---------")
            return
        if len(grid_path_length) != len(topopath_length):
            print("-----------Failed to find some path in topomap optmize part!-------------")
            return
        #compare
        #find topo_div_grid>1.5 and this path have shortest topo path
        topo_div_grid = topopath_length/grid_path_length
        tmp_grid_path = grid_path_length[(topo_div_grid > 1.5) & (topo_path_vertex_num > 4)  ]
        if len(tmp_grid_path) == 0:
            # dont find any shorter path
            return
        max_index = np.where(grid_path_length == min(tmp_grid_path))[0][0]
        max_path = np.array(calculate_grid_path.foundPath[max_index])
        max_path = np.fliplr(max_path)
        created_path = np.array(self.create_an_edge_between_two_vertex(max_path,now_global_map )) #(x,y)format path, n*2,include start and end
        created_path = created_path[1:-1]
        #add this path into topo map
        add_vertex_number = len(created_path)
        if add_vertex_number!=0:
            start_index = target_id_list[-1]
            end_index = max_index
            pose_in_map_frame = created_path*self.map_resolution + map_origin
            for i in range(add_vertex_number):
                now_pose = pose_in_map_frame[i]
                now_pose_list = list(now_pose)
                now_pose_list.append(0)
                vertex = Support_Vertex(self.self_robot_name, id=-1, pose=now_pose_list)
                self.last_vertex_id, self.current_node = self.map.add(vertex)
                self.map.vertex[-2].local_free_space_rect  = find_local_max_rect(self.global_map, self.map.vertex[-2].pose[0:2], self.map_origin, self.map_resolution)
                #add edge
                if i==0:
                    #connect with begin
                    link = [[self.self_robot_name, self.last_vertex_id], [self.self_robot_name, start_index]]
                    self.map.edge.append(Edge(id=self.map.edge_id, link=link))
                    self.map.edge_id += 1
                else:
                    link = [[self.self_robot_name, self.last_vertex_id], [self.self_robot_name, self.map.vertex[-2].id]]
                    self.map.edge.append(Edge(id=self.map.edge_id, link=link))
                    self.map.edge_id += 1

                if i == add_vertex_number - 1:
                    link = [[self.self_robot_name, self.last_vertex_id], [self.self_robot_name, end_index]]
                    self.map.edge.append(Edge(id=self.map.edge_id, link=link))
                    self.map.edge_id += 1

    
    def create_an_edge_between_two_vertex(self, path, global_map):
        # 计算path中间点的下标
        if len(path) <= 2:
            return []
        total_vertex = [path[0]]
        total_length = len(path) 
        start_index = 0
        
        while start_index < total_length - 1:
            mid_index = total_length - 1 
            while mid_index > start_index:
                if self.free_space_line_map(path[start_index],path[mid_index],global_map):
                    total_vertex.append(path[mid_index])
                    start_index = mid_index
                    break
                else:
                    mid_index = (mid_index - start_index)//2 + start_index
        
        return total_vertex


    def create_edge(self):
        #connect all edge with nearby vertex
        if len(self.map.vertex) == 1:
            return
        map_origin = np.array(self.map_origin)
        add_angle = []
        create_edge_num = 0
        for  i in range(len(self.map.vertex) - 2, -1, -1):
            now_vertex = self.map.vertex[i]
            last_vertex_pose = np.array(now_vertex.pose[0:2])
            now_vertex_pose = np.array(self.current_node.pose[0:2])
            if np.linalg.norm(last_vertex_pose - now_vertex_pose) < 20: # not too far away vertex; 
                last_vertex_pose_pixel = ( last_vertex_pose- map_origin)/self.map_resolution
                now_vertex_pose_pixel = (now_vertex_pose - map_origin)/self.map_resolution
                if self.free_space_line(last_vertex_pose_pixel, now_vertex_pose_pixel):

                    last_vertex_pose_center = last_vertex_pose - now_vertex_pose
                    last_vertex_angle = np.arctan2(last_vertex_pose_center[1],last_vertex_pose_center[0])
                    near_add_edge_flag = False
                    for old_angle in add_angle:
                        angle_tmp = last_vertex_angle - old_angle
                        angle_tmp = np.arctan2(np.sin(angle_tmp),np.cos(angle_tmp))
                        if abs(angle_tmp) < 0.5:
                            near_add_edge_flag = True
                            break
                    if not near_add_edge_flag:
                        create_edge_num +=1
                        add_angle.append(last_vertex_angle)
                        link = [[now_vertex.robot_name, now_vertex.id], [self.current_node.robot_name, self.current_node.id]]
                        self.map.edge.append(Edge(id=self.map.edge_id, link=link))
                        self.map.edge_id += 1
        
        if create_edge_num == 0:
            now_vertex = self.map.vertex[-2]
            link = [[now_vertex.robot_name, now_vertex.id], [self.current_node.robot_name, self.current_node.id]]
            self.map.edge.append(Edge(id=self.map.edge_id, link=link))
            self.map.edge_id += 1


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


    def change_goal(self):
        # move goal:now_pos + basic_length+offset;  now_angle + nextmove
        if len(self.total_frontier) == 0:
            return
        move_goal = self.choose_exp_goal()
        goal_message, self.goal = self.get_move_goal(self.self_robot_name,move_goal )#offset = 0
        goal_marker = self.get_goal_marker(self.self_robot_name, move_goal)
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)


    def map_grid_callback(self, data):
        
        #generate grid map and global grid map
        range = int(6/self.map_resolution)
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)
        timenow = rospy.Time.now()
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
        try:
            #detect frontier
            current_frontier = detect_frontier(self.global_map) * self.map_resolution + np.array(self.map_origin)
            self.total_frontier = np.vstack((self.total_frontier, current_frontier))
            ds_size = 0.2
            self.total_frontier = sparse_point_cloud(self.total_frontier, ds_size)
        except:
            pass
        self.grid_map_ready = 1
        self.update_frontier()

        if len(self.map.vertex) != 0:
            topomap_message = TopomapToMessage(self.map)
            self.topomap_pub.publish(topomap_message) # publish topomap important!

        if self.vertex_map_ready:
            self.visulize_vertex()
        #保存图片
        if save_result:
            temp = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
            temp[np.where(temp==-1)] = 125
            cv2.imwrite(debug_path+self.self_robot_name + "_local_map.jpg", temp)
            cv2.imwrite(debug_path+self.self_robot_name +"_global_map.jpg", self.global_map)


    def move_base_status_callback(self, data):
        try:
            status = data.status_list[-1].status
        # print(status)
        
            if status >= 3:
                self.erro_count +=1
            if self.erro_count >= 3:
                self.change_goal()
                self.erro_count = 0
        except:
            pass


    def get_move_goal(self, robot_name, goal)-> MoveBaseGoal():
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()

        # orientation = R.from_euler('z', move_direction, degrees=True).as_quat()
        # goal_message.target_pose.pose.orientation.x = orientation[0]
        # goal_message.target_pose.pose.orientation.y = orientation[1]
        # goal_message.target_pose.pose.orientation.z = orientation[2]
        # goal_message.target_pose.pose.orientation.w = orientation[3]
        # dont decide which orientation to choose 
        goal_message.target_pose.pose.orientation.x = 0
        goal_message.target_pose.pose.orientation.y = 0
        goal_message.target_pose.pose.orientation.z = 0
        goal_message.target_pose.pose.orientation.w = 1
        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_message.target_pose.pose.position = pose

        return goal_message, goal


    def get_goal_marker(self, robot_name, goal) -> PoseStamped():
        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        
        goal_marker.pose.orientation.x = 0
        goal_marker.pose.orientation.y = 0
        goal_marker.pose.orientation.z = 0
        goal_marker.pose.orientation.w = 1

        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]

        goal_marker.pose.position = pose

        return goal_marker


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
        #负责删除一部分前沿点
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
            if isinstance(vertex, Support_Vertex):
                continue
            if vertex.robot_name==self.self_robot_name or vertex.id in self.matched_vertex_dict[vertex.robot_name]:
                # already added vertex or vertex belong to this robot
                pass
            else:
                max_score = 0
                max_index = -1
                for index2, svertex in enumerate(self.map.vertex):#match vertex
                    if isinstance(svertex, Support_Vertex):
                        continue
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
                    local_laserscan_angle = now_matched_vertex.local_laserscan_angle
                    valid_indices = np.isfinite(local_laserscan_angle)
                    local_laserscan  = np.array(local_laserscan_angle * self.laser_scan_cos_sin)[:,valid_indices]
                    pc1 = np.vstack((local_laserscan, np.zeros(local_laserscan.shape[1],dtype=float)))
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
                if now_global_map[y, x] != 255:#排除掉经过unknown的部分
                    return False
        return True

    def free_space_line(self,point1,point2):
        # check whether a line cross a free space
        # point1 and point2 in pixel frame
        now_global_map = self.global_map
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