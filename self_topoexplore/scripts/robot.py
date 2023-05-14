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
from std_msgs.msg import Header

from robot_function import *
from RelaPose_2pc_function import *

debug_path = "/home/master/debug/"

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
        self.no_place_to_go = 0
        self.goal = np.array([10000.0, 10000.0])

        self.laserProjection = LaserProjection()
        self.pcd_queue = Queue(maxsize=10)# no used
        self.detect_loop = 0
        self.grid_map_ready = 0
        self.tf_transform_ready = 0
        self.cv_bridge = CvBridge()
        self.map_resolution = float(rospy.get_param('map_resolution', 0.01))
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.last_vertex = -1
        self.current_node = None
        self.last_nextmove = 0 #angle
        self.topomap_meet = 0
        self.vertex_dict = dict()
        self.edge_dict = dict()
        self.relative_position = dict()
        self.relative_orientation = dict()
        self.meeted_robot = list()
        for item in robot_list:
            self.vertex_dict[item] = list()
            self.edge_dict[item] = list()
            self.relative_position[item] = [0, 0, 0]
            self.relative_orientation[item] = 0


        self.topomap_matched_score = 0.65
        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        #relative pose estimation
        self.image_req_sub = rospy.Subscriber('/request_image', String, self.receive_image_req_callback)
        self.image_data_sub = None

        self.relative_pose_image = None
        self.image_ready = 0
        self.image_req_publisher = rospy.Publisher('/request_image', String, queue_size=1)
        self.image_data_pub = rospy.Publisher(robot_name+'/relative_pose_est_image', Image, queue_size=1)#发布自己节点的图片

        x_offset = 0.1
        y_offset = 0.2
        self.cam_trans = [[x_offset,0,0],[0,y_offset,math.pi/2],[-x_offset,0.0,math.pi],[0,-y_offset,-math.pi/2]] # camera position
        self.estimated_vertex_pose = list() #["robot_i","robot_j",id1,id2,estimated_pose] suppose that i < j
        self.map_frame_pose = dict() # map_frame_pose[robot_j] is [R_j,t_j] R_j 3x3
        self.ready_for_topo_map = True
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
        self.frontier_publisher = rospy.Publisher(robot_name+'/frontier_points', Marker, queue_size=10)
            

        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.loop_detect_callback, queue_size=1)#data is not important
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)

        for robot in robot_list:
            rospy.Subscriber(
                robot+"/topomap", TopoMapMsg, self.topomap_callback, queue_size=1)
            rospy.Subscriber(
                robot+"/ud", UnexploredDirectionsMsg, self.unexplored_directions_callback, robot, queue_size=1)
        
        rospy.Subscriber(
            robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)
        
        self.actoinclient.wait_for_server()



    def create_panoramic_callback(self, image1, image2, image3, image4):
        #合成一张全景图片然后发布
        img1 = self.cv_bridge.imgmsg_to_cv2(image1, desired_encoding="rgb8")
        img2 = self.cv_bridge.imgmsg_to_cv2(image2, desired_encoding="rgb8")
        img3 = self.cv_bridge.imgmsg_to_cv2(image3, desired_encoding="rgb8")
        img4 = self.cv_bridge.imgmsg_to_cv2(image4, desired_encoding="rgb8")
        panoram = [img1, img2, img3, img4]
        self.panoramic_view = np.hstack(panoram)
        # cv2.imwrite("/home/master/debug/panormaic.jpg", cv2.cvtColor(self.panoramic_view, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/1.jpg", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/2.jpg", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/3.jpg", cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("/home/master/debug/4.jpg", cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
        image_message = self.cv_bridge.cv2_to_imgmsg(self.panoramic_view, encoding="rgb8")
        image_message.header.stamp = rospy.Time.now()  
        image_message.header.frame_id = robot_name+"/odom"
        self.panoramic_view_pub.publish(image_message)

    def receive_image_callback(self, image):
        self.relative_pose_image = self.cv_bridge.imgmsg_to_cv2(image)
        self.image_ready = 1

    def receive_image_req_callback(self, req):
        #req: i_j_id
        input_list = req.data.split()  # 将字符串按照空格分割成一个字符串列表
        if input_list[1] in self.self_robot_name:
            # publish image msg
            image_index = int(input_list[2])
            req_image = self.map.vertex[image_index].local_image
            header = Header(stamp=rospy.Time.now())
            image_msg = self.cv_bridge.cv2_to_imgmsg(req_image, encoding="mono8")
            image_msg.header = header
            self.image_data_pub.publish(image_msg) #finish publish message
            print("robot = ",self.self_robot_name,"  publish image index = ",image_index)




    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        #goal pose offset
        offset = 0
        #init cvBridge
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)

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
        # ----finish getting pose----
        
        current_pose = copy.deepcopy(self.pose)
        vertex = Vertex(robot_name, id=-1, pose=current_pose, descriptor=feature, local_image=cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2GRAY))
        self.last_vertex, self.current_node, matched_flag = self.map.add(vertex, self.last_vertex, self.current_node)
        if matched_flag==0:# add a new vertex
            #create a new one
            while self.grid_map_ready==0 or self.tf_transform_ready==0:
                time.sleep(0.5)
            # 1. set and publish the topological map visulization markers
            localMap = self.grid_map
            self.map.vertex[-1].localMap = localMap #every vertex contains local map
            self.detect_loop = 0
            picked_vertex_id = self.map.upgradeFrontierPoints(self.last_vertex,resolution=self.map_resolution)

            marker_array = MarkerArray()
            marker_message = set_marker(robot_name, len(self.map.vertex), self.map.vertex[0].pose, action=Marker.DELETEALL)
            marker_array.markers.append(marker_message)
            self.marker_pub.publish(marker_array) #DELETEALL 操作，防止重影
            

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

            
            # 2. choose navigableDirection
            navigableDirection = self.map.vertex[picked_vertex_id].navigableDirection
            nextmove = 0
            directionID = 0
            max_dis = 0
            dis_with_other_centers = [0 for i in range(len(self.meeted_robot))]
            dis_scores = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
            dis_with_vertices = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
            epos = 0.2

            if len(navigableDirection) == 0:#no navigable Dirention
                self.no_place_to_go = 1
            else:
                # choose where to go
                for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints)):
                    position_tmp = self.map.vertex[picked_vertex_id].frontierPoints[i]
                    now_vertex_pose = np.array(self.map.vertex[picked_vertex_id].pose[0:2])
                    # map.center : center of vertex
                    dis_tmp = np.sqrt(np.sum(np.square(position_tmp + now_vertex_pose - self.map.center)))#the point farthest from the center of the map
                    dis_scores[i] += epos * dis_tmp
                    #TODO
                    #需要修改这一部分
                    for j in range(len(self.meeted_robot)):# remain considering, deal with multi robot part
                        dis_with_other_centers[j] = np.sqrt(np.sum(np.square(position_tmp-self.map.center_dict[self.meeted_robot[j]])))
                        dis_scores[i] += dis_with_other_centers[j] * (1-epos)

                #choose max dis as next move
                for i in range(len(dis_scores)):
                    if dis_scores[i] > max_dis:
                        max_dis = dis_scores[i]
                        directionID = i
                move_direction = navigableDirection[directionID]
                
                # delete this frontier
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
                # self.actoinclient.send_goal(goal_message)
                self.goal_pub.publish(goal_marker)

        #deal with no place to go
        if self.no_place_to_go:
            find_a_goal = 0
            position = self.pose
            position = np.array([position[0], position[1]])
            distance_list = []
            for i in range(len(self.map.vertex)):
                temp_position = self.map.vertex[i].pose
                temp_position = np.asarray([temp_position[0], temp_position[1]])
                distance_list.append(np.linalg.norm(temp_position - position))
            while(distance_list):
                min_dis = min(distance_list)
                index = distance_list.index(min_dis)
                if len(self.map.vertex[index].navigableDirection) != 0:
                    nextmove = self.map.vertex[index].navigableDirection[0]
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
            self.no_place_to_go = 0



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
            
            self.global_map = np.asarray(data.data).reshape(shape)
            #获取当前一个小范围的grid map
            self.grid_map = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
            self.grid_map[np.where(self.grid_map==-1)] = 255
            if robot_name == 'robot1':
                self.global_map[np.where(self.global_map==-1)] = 255
                temp = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
                temp[np.where(temp==-1)] = 125
                # cv2.imwrite(debug_path+"map.jpg", temp)
                # cv2.imwrite(debug_path+"/globalmap.jpg", self.global_map)
            self.grid_map_ready = 1
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
                self.no_place_to_go = 1
                self.erro_count = 0
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

    def loop_detect_callback(self, data):
        position = self.pose
        position = np.array([position[0], position[1]])
        if self.detect_loop:
            #delete unexplored direction based on distance between now robot pose and frontier point position
            for i in range(len(self.map.vertex)):
                meeted = list()
                vertex = self.map.vertex[i]
                now_vertex_pose = np.array([vertex.pose[0],vertex.pose[1]])

                for j in range(len(vertex.navigableDirection)-1, -1, -1):
                    unexplored = vertex.frontierPoints[j] + now_vertex_pose # change into map frame

                    if np.sqrt(np.sum(np.square(position-unexplored))) < 3:#delete unexplored position
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
            if self.reference_vertex == None:
                self.reference_vertex = self.current_node
            rr = int(6/self.map_resolution)
            for i in range(len(self.map.vertex)):
                vertex = self.map.vertex[i]
                vertex_position = np.array([vertex.pose[0], vertex.pose[1]])
                if vertex.navigableDirection:
                    if np.sqrt(np.sum(np.square(position-vertex_position))) < 100:
                        location = [0, 0]
                        shape = self.global_map.shape
                        location[0] = self.current_loc_pixel[0] + int((self.pose[0] - vertex_position[0])/self.map_resolution)
                        location[1] = self.current_loc_pixel[1] - int((self.pose[1] - vertex_position[1])/self.map_resolution)
                        temp_map = self.global_map[max(location[0]-rr,0):min(location[0]+rr,shape[0]), max(location[1]-rr,0):min(location[1]+rr, shape[1])]
                        self.map.vertex[i].localMap = temp_map #renew local map
                        old_ud = self.map.vertex[i].navigableDirection
                        new_ud = self.map.vertex[i].navigableDirection
                        delete_list = []
                        for j in range(len(old_ud)):
                            not_deleted = 1
                            for uds in new_ud:
                                if abs(old_ud[j]-uds) < 5:
                                    not_deleted = 0
                            if not_deleted == 1:
                                delete_list.append(j)
                        for uds in delete_list:
                            ud_message = UnexploredDirectionsMsg()
                            ud_message.robot_name = self.map.vertex[i].robot_name
                            ud_message.vertexID = self.map.vertex[i].id
                            ud_message.directionID = uds
                            self.unexplore_direction_pub.publish(ud_message)

                if np.linalg.norm(position-vertex_position) < 2.5:#现在机器人位置距离节点位置很近
                    new_vertex_on_path = 1
                    for svertex in self.vertex_on_path:
                        if svertex.robot_name == vertex.robot_name and svertex.id == vertex.id:
                            new_vertex_on_path = 0
                    if new_vertex_on_path == 1 and vertex.robot_name!= self.reference_vertex.robot_name and vertex.id != self.reference_vertex.id:
                        self.vertex_on_path.append(vertex)
                if np.linalg.norm(self.goal - vertex_position) < 3:
                    self.vertex_on_path.append(vertex)

        if len(self.map.vertex) != 0:
            topomap_message = TopomapToMessage(self.map)
            self.topomap_pub.publish(topomap_message) # publish topomap important!
        
        if len(self.vertex_on_path) >= 3:
            self.no_place_to_go = 1
            print(robot_name, "no place to go")
            self.vertex_on_path = []
            self.reference_vertex = self.current_node

    def topomap_callback(self, topomap_message):
        # receive topomap from other robots
        if not self.ready_for_topo_map:
            return
        self.ready_for_topo_map = False
        Topomap = MessageToTopomap(topomap_message)
        matched_vertex = list()
        vertex_pair = dict()
        for vertex in Topomap.vertex:
            if vertex.robot_name not in self.map.center_dict.keys():#estimated /robot_j/map
                self.map.center_dict[vertex.robot_name] = np.array([0.0, 0.0])
            if vertex.robot_name not in self.vertex_dict.keys():
                self.vertex_dict[vertex.robot_name] = list() # init vertex of other robot
            elif (vertex.id in self.vertex_dict[vertex.robot_name]) or (vertex.robot_name==robot_name):
                # already added vertex or vertex belong to this robot
                pass
            else:
                for svertex in self.map.vertex:#match vertex
                    if svertex.robot_name == vertex.robot_name:#created by same robot
                        pass
                    else:
                        score = np.dot(vertex.descriptor.T, svertex.descriptor)
                        if score > self.topomap_matched_score:
                            print("matched:", svertex.robot_name, svertex.id, vertex.robot_name, vertex.id, score)
                            #estimate relative position
                            #请求一下图片
                            self.image_data_sub  = rospy.Subscriber(vertex.robot_name+"/relative_pose_est_image", Image, self.receive_image_callback)
                            req_string = self.self_robot_name[-1] +" "+vertex.robot_name[-1]+" "+str(vertex.id)
                            while not self.image_ready:
                                # 发布请求图片的消息
                                self.image_req_publisher.publish(req_string)
                                rospy.sleep(0.1)
                            #unsubscrib image topic 
                            self.image_data_sub.unregister()
                            self.image_data_sub = None

                            img1 = svertex.local_image
                            img2 = self.relative_pose_image #from other robot

                            #estimated pose
                            pose = planar_motion_calcu_mulit(img1,img2,k1 = self.K_mat,k2 = self.K_mat,cam_pose = self.cam_trans)
                            
                            self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.id,vertex.id,pose])

                            matched_vertex.append(vertex)
                            vertex_pair[vertex.id] = svertex.id

                            #add vertex
                            tmp_link = [[svertex.robot_name, svertex.id], [vertex.robot_name, vertex.id]]
                            now_edge = self.edge.append(Edge(id=self.map.edge_id, link=tmp_link))
                            self.map.edge_id += 1
                            self.map.edge.append(now_edge)
                            break

        for edge in Topomap.edge:
            if edge.link[0][1] in self.vertex_dict[edge.link[0][0]]:
                if edge.link[1][1] in self.vertex_dict[edge.link[1][0]] or edge.link[1][0]==self.self_robot_name:
                    pass
                else:
                    edge.link[0][0] = self.self_robot_name
                    edge.link[0][1] = vertex_pair[edge.link[0][1]]
                    edge.id = self.map.edge_id
                    self.map.edge_id += 1
                    self.map.edge.append(edge)
            elif edge.link[1][1] in self.vertex_dict[edge.link[1][0]]:
                if edge.link[0][1] in self.vertex_dict[edge.link[0][0]] or edge.link[0][0]==self.self_robot_name:
                    pass
                else:
                    edge.link[1][0] = self.self_robot_name
                    edge.link[1][1] = vertex_pair[edge.link[1][1]]
                    edge.id = self.map.edge_id
                    self.map.edge_id += 1
                    self.map.edge.append(edge)
            else:
                edge.id = self.map.edge_id
                self.map.edge_id += 1
                self.map.edge.append(edge)

        #estimate the center of other robot
        #TODO
        self.topo_optimize()

        for vertex in Topomap.vertex:
            #add vertex
            if vertex.id not in self.vertex_dict[vertex.robot_name] and vertex.robot_name != self.self_robot_name:
                now_robot_map_frame_rot = self.map_frame_pose[vertex.robot_name][0]
                now_robot_map_frame_trans = self.map_frame_pose[vertex.robot_name][1]
                tmp_pose = now_robot_map_frame_rot @ np.array([vertex.pose[0],vertex.pose[1],0]) + now_robot_map_frame_trans
                vertex.pose[0] = tmp_pose[0]
                vertex.pose[1] = tmp_pose[1]
                self.map.vertex.append(vertex)
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
        #TODO
        #self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.id,vertex.id,pose])
        # This part should update self.map_frame_pose[vertex.robot_name];self.map_frame_pose[vertex.robot_name][0] R33;[1]t 31
        pass

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