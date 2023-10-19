#!/usr/bin/python3.8
from tkinter.constants import Y
import rospy
from rospy.timer import Rate, sleep
from sensor_msgs.msg import Image, LaserScan
import rospkg
import tf
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, PoseStamped, Point
from laser_geometry import LaserProjection
import message_filters
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from self_topoexplore.msg import TopoMapMsg
from self_topoexplore.msg import ImageWithPointCloudMsg
from TopoMap import Support_Vertex, Vertex, Edge, TopologicalMap
from utils.imageretrieval.imageretrievalnet import init_network
from utils.pointnet_model import PointNet_est
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
        path = rospack.get_path('multi_fht_map')
        # in simulation environment each robot has same intrinsic matrix
        self.K_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))
        #network part
        network = rospy.get_param("~network")
        self.network_gpu = rospy.get_param("~platform")
        if network in PRETRAINED:
            state = load_url(PRETRAINED[network], model_dir= os.path.join(path, "data/networks"))
        else:
            state = torch.load(network)
        torch.cuda.empty_cache()
        current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        point_net_path = os.path.join(current_path,'../data/networks', 'my_pointnet_20231016151924.pth') # Load PointNet
        self.pointnet =  PointNet_est(k=3)
        self.pointnet.load_state_dict(torch.load(point_net_path))  #load param of network
        self.pointnet.eval()
        net_params = get_net_param(state)
        self.net = init_network(net_params)#Load Image Retrivial
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
        self.adj_list = dict()#拓扑地图到邻接矩阵
        self.meeted_robot = list()
        self.potential_main_vertex = list()
        for item in robot_list:
            self.vertex_dict[item] = list()
            self.matched_vertex_dict[item] = list() #和其他机器人已经匹配上的节点

        #For multi robot map merge
        self.topomap_matched_score = 0.96
        
        self.topomap_dict = dict() #FHT-Map of other map
        self.topomap_robot1frame_dict = dict() #FHT-Map of other map in robot1 frame
        self.topomap_robot1frame_dict[robot_name] = None
        if robot_name == 'robot1':
            for item in robot_list:
                self.topomap_dict[item] = None
                self.topomap_robot1frame_dict[item] = None

        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        #relative pose estimation
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
        if robot_name != "robot1":#other robot publish topomap
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
        #only robot1 subscribe all vertex
        if robot_name == "robot1":
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
        
        # valid_indices = np.isfinite(ranges)
        # self.local_laserscan  = np.array(ranges[valid_indices] * self.laser_scan_cos_sin[:,valid_indices])
        # self.local_laserscan  = np.array(ranges * self.laser_scan_cos_sin)
        # a = np.sum(self.local_laserscan[:,valid_indices]**2,axis=0)**0.5
        self.local_laserscan_angle = ranges


    def heat_value_eval(self):
        # evluate now laser scan data using point net
        local_laserscan_angle = copy.deepcopy(self.local_laserscan_angle )
        valid_indices = np.isfinite(local_laserscan_angle)
        local_laserscan_angle[~valid_indices] = 8
        local_laserscan  = np.array(local_laserscan_angle* self.laser_scan_cos_sin)
        now_laserscan = np.vstack((local_laserscan,np.zeros(local_laserscan.shape[1])))
        lidar_data_torch = torch.tensor(now_laserscan).float()
        lidar_data_torch = lidar_data_torch.view((-1,3,360))
        self.pointnet.to('cpu')
        output = self.pointnet(lidar_data_torch).detach().numpy().reshape(-1)[0]

        return output

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



    def visulize_vertex(self):
        #对self.topomap_robot1frame_dict[robot_name]中所有做可视化
        self.topomap_robot1frame_dict[robot_name] = self.map
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
        for now_map in self.topomap_robot1frame_dict.values():
            if now_map == None:
                continue
            ori_map = R.from_matrix(now_map.rotation).as_quat()
            for index, now_vertex in enumerate(now_map.vertex):
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
                marker.pose.orientation.x = ori_map[0]
                marker.pose.orientation.y = ori_map[1]
                marker.pose.orientation.z = ori_map[2]
                marker.pose.orientation.w = ori_map[3]
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
        markerid = 0
        main_vertex_color = (self.vis_color[1][0], self.vis_color[1][1], self.vis_color[1][2])
        support_vertex_color = (self.vis_color[2][0], self.vis_color[2][1], self.vis_color[2][2])
        for now_map in self.topomap_robot1frame_dict.values():
            if now_map == None:
                continue
            for vertex in now_map.vertex:
                if isinstance(vertex, Vertex):
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=main_vertex_color, scale=0.5)
                else:
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=support_vertex_color, scale=0.4)
                
                marker_array.markers.append(marker_message)
                markerid += 1
        
        #visualize edge
        #可视化edge就是把两个vertex的pose做一个连线
        main_edge_color = (self.vis_color[3][0], self.vis_color[3][1], self.vis_color[3][2])
        edge_array = MarkerArray()
        now_edge_id = 0
        for now_map in self.topomap_robot1frame_dict.values():
            if now_map == None:
                continue
            for edge in now_map.edge:
                poses = []
                poses.append(now_map.vertex[edge.link[0][1]].pose)
                poses.append(now_map.vertex[edge.link[1][1]].pose)
                edge_message = set_edge(robot_name, now_edge_id, poses, "edge",main_edge_color, scale=0.1,frame_name = "/map")
                now_edge_id += 1
                edge_array.markers.append(edge_message)
        
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
    
    def finish_exploration(self):
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
    
    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        self.update_robot_pose() #update robot pose
        
        current_pose = copy.deepcopy(self.pose)
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        
        
        #check whether the robot stop moving
        if not self.finish_explore and len(self.total_frontier) == 0: 
            self.finish_exploration()
            
        create_a_vertex_flag = self.create_a_vertex(panoramic_view) # whether create a vertex
        if create_a_vertex_flag: # create a vertex
            if create_a_vertex_flag == 1:#create a main vertex
                omega_ch = np.array([1,2]) 
                ch_list = []
                for  now_vertex in self.potential_main_vertex:
                    C_now = now_vertex[1][0]
                    H_now = now_vertex[1][1]
                    ch_list.append([C_now, H_now])
                
                total_ch = np.array(ch_list)
                z_star = np.max(total_ch,axis=0)
                total_ch_minus_z = total_ch - z_star
                weighted_ch = omega_ch * total_ch_minus_z
                infinite_norm_weighted_ch = np.max(weighted_ch,axis=1)
                best_index = np.argmax(infinite_norm_weighted_ch)
                vertex = copy.deepcopy(self.potential_main_vertex[best_index][0])
                self.last_vertex_id, self.current_node = self.map.add(vertex)
                self.potential_main_vertex = []
                
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
            
            if create_a_vertex_flag ==1 or create_a_vertex_flag ==2: 
                refine_topo_map_msg = String()
                refine_topo_map_msg.data = "Start_find_path!"
                self.find_better_path_pub.publish(refine_topo_map_msg) #find a better path
        
    def create_a_vertex(self,panoramic_view):
        #return 1 for uncertainty value reach th; 2 for not a free space line
        #and 0 for don't creat a vertex

        #important parameter
        feature_simliar_th = 0.94
        main_vertex_dens = 25 #main_vertex_dens^0.5 is the average distance of a vertex, 4 is good; sigma_c in paper
        global_vertex_dens = 2 # create a support vertex large than 2 meter
        
        # check the heat map value of this point
        local_laserscan_angle = copy.deepcopy(self.local_laserscan_angle)
        heat_map_value = self.heat_value_eval()
        now_feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)

        max_sim = 0
        C_now = 0#calculate C
        now_pose = np.array(self.pose[0:2])
        for now_vertex in self.map.vertex:
            if isinstance(now_vertex, Support_Vertex):
                continue
            now_similarity = np.dot(now_feature.T, now_vertex.descriptor)
            max_sim = max(now_similarity,max_sim)
            now_vertex_pose = np.array(now_vertex.pose[0:2])
            dis = np.linalg.norm(now_vertex_pose - now_pose)
            # print(now_vertex.descriptor_infor)
            C_now += now_vertex.descriptor_infor * np.exp(-dis**2 / main_vertex_dens)

        #similarity smaller than th
        if max_sim < feature_simliar_th:          
            #calculate H
            H_now = heat_map_value
            
            #判断是否位于parote optimal front
            #self.potential_main_vertex: [[vertex,[C,H]],...]
            remove_list = []
            on_pareto_optimal_front_flag = True
            for index, now_vertex in enumerate(self.potential_main_vertex):
                old_C = now_vertex[1][0]
                old_H = now_vertex[1][1]

                #C: 取min; H： 取max
                if old_C < C_now and old_H > H_now:#dominated by old vertex
                    on_pareto_optimal_front_flag = False
                    break
                if old_C > C_now and old_H < H_now:#dominates old vertex
                    remove_list.append(index)
            if on_pareto_optimal_front_flag:#更新最优点
                new_potential_vertex = [value for i, value in enumerate(self.potential_main_vertex) if i not in remove_list]
                gray_local_img = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2GRAY)
                vertex = Vertex(robot_name, id=-1, pose=copy.deepcopy(self.pose), descriptor=copy.deepcopy(now_feature), local_image=gray_local_img, local_laserscan_angle=local_laserscan_angle)
                new_potential_vertex.append([vertex,[C_now,H_now]])
                self.potential_main_vertex = new_potential_vertex
            
            # print("len of potential vertex list:", len(self.potential_main_vertex))
        
        if C_now < 0.368:
            # create a main vertex
            if len(self.potential_main_vertex) != 0:
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
                free_line_flag = self.expanded_free_space_line(last_vertex_pose_pixel, now_robot_pose, 5)
            
            if free_line_flag:
                break
        
        if not free_line_flag:#if not a line in free space, create a support vertex
            return 2

        min_dens_flag = False
        for last_vertex in self.map.vertex:
            last_vertex_pose = np.array(last_vertex.pose[0:2])
            if np.linalg.norm(now_pose - last_vertex_pose) < global_vertex_dens:
                min_dens_flag = True

        if not min_dens_flag:#if robot in a place with not that much vertex, then create a support vertex
            return 3
        
        return 0

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
        now_global_map = copy.deepcopy(self.global_map)
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
            if np.linalg.norm(last_vertex_pose - now_vertex_pose) < 8: # not too far away vertex
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
        #可能会存在bug
        if create_edge_num == 0:
            #如果出现一个悬空节点，直接连接他和最近的节点
            min_vertex = 0
            min_dis = 1e10
            for  i in range(len(self.map.vertex) - 2, -1, -1):
                now_vertex = self.map.vertex[i]
                last_vertex_pose = np.array(now_vertex.pose[0:2])
                now_vertex_pose = np.array(self.current_node.pose[0:2])
                if np.linalg.norm(last_vertex_pose - now_vertex_pose) < min_dis:
                    min_dis = np.linalg.norm(last_vertex_pose - now_vertex_pose)
                    min_vertex = i
            now_vertex = self.map.vertex[min_vertex]
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
            if self.self_robot_name != 'robot1':    
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
        # 原始的topomap储存在self.topomap_dict下
        # 变换过相对位姿关系的topomap储存在self.topomap_robot1frame_dict下

        update_robot_center = False
        self.ready_for_topo_map = False
        Topomap = MessageToTopomap(topomap_message)
        if len(Topomap.vertex) ==0: # don't have any vertex in this map
            return
        now_robot_name = Topomap.vertex[0].robot_name
        self.topomap_dict[now_robot_name] = copy.deepcopy(Topomap)
        

        # find max matched vertex
        for vertex in Topomap.vertex:
            if isinstance(vertex, Support_Vertex):
                continue
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

            if max_score > 0:
                final_R, final_t = self.single_estimation(vertex,self.map.vertex[max_index])
                if final_R is None or final_t is None:
                    return
                else:
                    estimated_pose = [final_t[0][0],final_t[1][0], math.atan2(final_R[1,0],final_R[0,0])/math.pi*180]
                    if vertex.id not in self.matched_vertex_dict[vertex.robot_name]:
                        self.matched_vertex_dict[vertex.robot_name].append(vertex.id) 
                        # pose optimize
                        self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,self.map.vertex[max_index].pose,list(vertex.pose),estimated_pose])
                        if now_robot_name not in self.meeted_robot:
                            self.meeted_robot.append(now_robot_name)
                        print("--------TOPO OPT----------")
                        self.topo_optimize()
                        # self.update_vertex_pose()
                        self.init_map_pose = True
                        
        #保存转换过相对位姿的两个地图
        if now_robot_name in self.map_frame_pose.keys():
            #perform map merge
            tmp_topomap = copy.deepcopy(Topomap)
            #change frame
            tmp_topomap.change_topomap_frame(self.map_frame_pose[now_robot_name]) #转换两个地图
            self.topomap_robot1frame_dict[now_robot_name] = tmp_topomap
        

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
    
    def single_estimation(self,vertex1,vertex2):
        #vertex1: received map 
        #vertex2: vertex of robot 1
        #return a 3x3 rotation matrix and a 3x1 tranform vector
        vertex_laser = vertex1.local_laserscan_angle
        now_laser = vertex2.local_laserscan_angle
        #do ICP to recover the relative pose
        now_laser_xy = self.angle_laser_to_xy(now_laser)
        vertex_laser_xy = self.angle_laser_to_xy(vertex_laser)

        pc1 = np.vstack((now_laser_xy, np.zeros(now_laser_xy.shape[1])))
        pc2 = np.vstack((vertex_laser_xy, np.zeros(vertex_laser_xy.shape[1])))

        processed_source = o3d.geometry.PointCloud()
        pc2_offset = copy.deepcopy(pc2)
        pc2_offset[2,:] -= 0.1
        processed_source.points = o3d.utility.Vector3dVector(np.vstack([pc2.T,pc2_offset.T]))

        processed_target = o3d.geometry.PointCloud()
        pc1_offset = copy.deepcopy(pc1)
        pc1_offset[2,:] -= 0.1
        processed_target.points = o3d.utility.Vector3dVector(np.vstack([pc1.T,pc1_offset.T]))

        final_R, final_t = ransac_icp(processed_source, processed_target, None, vis=0)

        return final_R, final_t


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
            print("---------------Robot Center Optimized-----------------------")
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
                # if now_global_map[y, x] != 255:#排除掉经过unknown的部分
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