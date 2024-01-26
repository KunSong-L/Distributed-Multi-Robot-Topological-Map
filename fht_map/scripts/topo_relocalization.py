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
        path = rospack.get_path('fht_map')

        self.th_match =  float(rospy.get_param("~th_match"))
        print(f"th_match is {self.th_match}")

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
        self.init_map_pose = False
        self.navigated_point = np.array([],dtype=float).reshape((-1,3))
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        #relative pose estimation
        x_offset = 0.1
        y_offset = 0.2
        self.cam_trans = [[x_offset,0,0],[0,y_offset,math.pi/2],[-x_offset,0.0,math.pi],[0,-y_offset,-math.pi/2]] # camera position
        self.estimated_vertex_pose = list() #["robot_i","robot_j",id1,id2,estimated_pose] suppose that i < j
        self.map_frame_pose = dict() # map_frame_pose[robot_j] is [R_j,t_j] R_j 3x3
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
        self.first_gen_topomap = True


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

    
    def convert_numpy_2_pointcloud2_color(self, points, stamp=None, frame_id=None, maxDistColor=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array of points. 
        This function will automatically assign RGB values to each point. The RGB values are
        determined by the distance of a point from the origin. Use maxDistColor to set the distance 
        at which the color corresponds to the farthest distance is used.
        points: A NumPy array of Nx3.
        stamp: An alternative ROS header stamp.
        frame_id: The frame id. String.
        maxDisColor: Should be positive if specified..
        '''
        
        # Clipping input.
        # dist = np.linalg.norm( points, axis=1 )
        dist = points[:,2]
        if ( maxDistColor is not None and maxDistColor > 0):
            dist = np.clip(dist, 0, maxDistColor)

        # Compose color.
        DIST_COLORS = [\
            "#2980b9",\
            "#27ae60",\
            "#f39c12",\
            "#c0392b",\
            ]

        DIST_COLOR_LEVELS = 50
        cr, cg, cb = color_map( dist, DIST_COLORS, DIST_COLOR_LEVELS )

        C = np.zeros((cr.size, 4), dtype=np.uint8) + 255

        C[:, 0] = cb.astype(np.uint8)
        C[:, 1] = cg.astype(np.uint8)
        C[:, 2] = cr.astype(np.uint8)

        C = C.view("uint32")

        # Structured array.
        pointsColor = np.zeros( (points.shape[0], 1), \
            dtype={ 
                "names": ( "x", "y", "z", "rgba" ), 
                "formats": ( "f4", "f4", "f4", "u4" )} )

        points = points.astype(np.float32)

        pointsColor["x"] = points[:, 0].reshape((-1, 1))
        pointsColor["y"] = points[:, 1].reshape((-1, 1))
        pointsColor["z"] = points[:, 2].reshape((-1, 1))
        pointsColor["rgba"] = C

        header = Header()

        if stamp is None:
            header.stamp = rospy.Time().now()
        else:
            header.stamp = stamp

        if frame_id is None:
            header.frame_id = "None"
        else:
            header.frame_id = frame_id

        msg = PointCloud2()
        msg.header = header

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width  = points.shape[0]

        msg.fields = [
            PointField('x',  0, PointField.FLOAT32, 1),
            PointField('y',  4, PointField.FLOAT32, 1),
            PointField('z',  8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
            ]

        msg.is_bigendian = False
        msg.point_step   = 16
        msg.row_step     = msg.point_step * points.shape[0]
        msg.is_dense     = int( np.isfinite(points).all() )
        msg.data         = pointsColor.tostring()

        return msg

    def publish_point_cloud(self):
        # 初始化ROS节点
        # 创建PointCloud2消息对象
        if len(self.navigated_point) <2:
            return
        pc_msg = self.convert_numpy_2_pointcloud2_color(self.navigated_point, stamp=None, frame_id=robot_name + "/map", maxDistColor=None)
        # 创建PointCloud2消息发布者
        self.pc_pub.publish(pc_msg)
    

    def visulize_vertex(self):
        # ----------visualize frontier------------
        
        # frontier_marker = Marker()
        # now = rospy.Time.now()
        # frontier_marker.header.frame_id = robot_name + "/map"
        # frontier_marker.header.stamp = now
        # frontier_marker.ns = "frontier_point"
        # frontier_marker.type = Marker.POINTS
        # frontier_marker.action = Marker.ADD
        # frontier_marker.pose.orientation.w = 1.0
        # frontier_marker.scale.x = 0.1
        # frontier_marker.scale.y = 0.1
        # frontier_marker.color.r = self.vis_color[0][0]
        # frontier_marker.color.g = self.vis_color[0][1]
        # frontier_marker.color.b = self.vis_color[0][2]
        # frontier_marker.color.a = 0.7
        # for frontier in self.total_frontier:
        #     point_msg = Point()
        #     point_msg.x = frontier[0]
        #     point_msg.y = frontier[1]
        #     point_msg.z = 0.2
        #     frontier_marker.points.append(point_msg)
        # self.frontier_publisher.publish(frontier_marker)
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
        if self.first_gen_topomap:
            marker_array = MarkerArray()
            markerid = 0
            main_vertex_color = (self.vis_color[1][0], self.vis_color[1][1], self.vis_color[1][2])
            support_vertex_color = (self.vis_color[2][0], self.vis_color[2][1], self.vis_color[2][2])
            for vertex in self.map.vertex:
                if vertex.robot_name != robot_name:
                    marker_message = set_marker(robot_name, markerid, vertex.pose)#other color
                else:
                    if isinstance(vertex, Vertex):
                        marker_message = set_marker(robot_name, markerid, vertex.pose, color=main_vertex_color, scale=0.5,frame_name = "/map_origin")
                    else:
                        marker_message = set_marker(robot_name, markerid, vertex.pose, color=support_vertex_color, scale=0.4,frame_name = "/map_origin")
                marker_array.markers.append(marker_message)
                markerid += 1
            #visualize edge
            #可视化edge就是把两个vertex的pose做一个连线
            main_edge_color = (self.vis_color[3][0], self.vis_color[3][1], self.vis_color[3][2])
            edge_array = MarkerArray()
            for edge in self.map.edge:
                poses = []
                poses.append(self.map.vertex[edge.link[0][1]].pose)
                poses.append(self.map.vertex[edge.link[1][1]].pose)
                edge_message = set_edge(robot_name, edge.id, poses, "edge",main_edge_color, scale=0.1,frame_name = "/map_origin")
                edge_array.markers.append(edge_message)
            
            self.marker_array = marker_array
            self.edge_array = edge_array
            self.first_gen_topomap = False
        
        #update time
        for i in range(len(self.marker_array.markers)):
            self.marker_array.markers[i].header.stamp = rospy.Time.now()
        for i in range(len(self.edge_array.markers)):
            self.edge_array.markers[i].header.stamp = rospy.Time.now()
        self.marker_pub.publish(self.marker_array)
        self.edge_pub.publish(self.edge_array)

    def choose_nav_goal(self):
        dis_epos = 1
        angle_epos = 2
        
        frontier_poses = self.total_frontier  

        dis_frontier_poses = np.sqrt(np.sum(np.square(frontier_poses - self.pose[0:2]), axis=1))
        dis_tmp = np.exp(-(dis_frontier_poses-3)**2 / 8)

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

    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        self.update_robot_pose() #update robot pose
        now_laser = copy.deepcopy(self.local_laserscan_angle)
        current_pose = copy.deepcopy(self.pose)
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
        information = calculate_entropy(feature)
        if not self.receive_topomap:
            return

        if self.init_map_pose:
            self.update_relative_pose()
            self.vertex_map_ready = True
        
        self.update_robot_pose() #update robot pose
        
        # init map pose
        best_match_rate = 0
        best_match_index = 0
        for index, now_vertex in enumerate(self.received_map.vertex):
            if isinstance(now_vertex, Support_Vertex):
                continue
            now_feature = now_vertex.descriptor
            now_match = np.dot(feature, now_feature)
            if now_match > best_match_rate:
                best_match_rate = now_match
                best_match_index = index
        
        # print("best match ratio = ",best_match_rate)

        self.navigated_point = np.vstack((self.navigated_point, np.array([current_pose[0],current_pose[1],best_match_rate])))
        self.navigated_point = sparse_point_cloud(self.navigated_point, 0.1)
        self.publish_point_cloud()
        if best_match_rate > self.th_match:
            vertex_laser = self.received_map.vertex[best_match_index].local_laserscan_angle
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

            if final_R is None or final_t is None:
                return
            else:
                estimated_pose = [final_t[0][0],final_t[1][0], math.atan2(final_R[1,0],final_R[0,0])/math.pi*180]
                if best_match_index not in self.matched_vertex:
                    self.matched_vertex.append(best_match_index) 
                    # pose optimize
                    self.estimated_vertex_pose.append([self.self_robot_name, "robot2",current_pose,list(self.received_map.vertex[best_match_index].pose),estimated_pose])
                    self.topo_optimize()
                    # self.update_vertex_pose()
                    self.update_relative_pose()
                    self.init_map_pose = True
   

    def change_goal(self):
        # move goal:now_pos + basic_length+offset;  now_angle + nextmove
        move_goal = self.choose_nav_goal()
        goal_message, self.goal = self.get_move_goal(self.self_robot_name,move_goal )#offset = 0
        goal_marker = self.get_goal_marker(self.self_robot_name, move_goal)
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)

    # def topo_optimize(self):
    #     #self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.pose,vertex.pose,pose])
    #     # This part should update self.map_frame_pose[vertex.robot_name];self.map_frame_pose[vertex.robot_name][0] R33;[1]t 31
    #     input = self.estimated_vertex_pose
    #     now_meeted_robot_num = len(self.meeted_robot)
    #     name_list = [self.self_robot_name] + self.meeted_robot
    #     c_real = [[0,0,0] for i in range(now_meeted_robot_num + 1)]

    #     now_id = 1
    #     trans_data = ""
    #     outlier_rejected_input = outlier_rejection(input)
    #     for center in c_real:
    #         trans_data+="VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n".format(now_id,center[0],center[1],center[2]/180*math.pi)
    #         now_id +=1

    #     for now_input in outlier_rejected_input:
    #         # pose_origin1 = numpy.append(pose_origin1, numpy.array([[now_input[2][0]],[now_input[2][1]]]), axis=1)
    #         # pose_origin2 = numpy.append(pose_origin2, numpy.array([[now_input[3][0]],[now_input[3][1]]]), axis=1)
    #         now_trust = 1
    #         start_idx = str(name_list.index(now_input[0])+1)
    #         end_idx = str(name_list.index(now_input[1])+1)
    #         trans_data+="EDGE_SE2 {} {} ".format(start_idx,end_idx)
    #         for j in range(3):
    #             for k in range(3):
    #                 trans_data += " {:.6f} ".format(now_input[2+j][k])
    #         trans_data += " {:.6f} 0 0 {:.6f} 0 {:.6f}\n".format(now_trust,now_trust,now_trust)
    #         now_id += 2

    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     # 构建可执行文件的相对路径
    #     executable_path = os.path.join(current_dir, '..', 'src', 'pose_graph_opt', 'pose_graph_2d')
    #     process = subprocess.Popen(executable_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    #     # 向C++程序输入数据
    #     process.stdin.write(trans_data)
    #     # 关闭输入流
    #     process.stdin.close()
    #     output_data = process.stdout.read()
    #     # 等待C++程序退出
    #     process.wait()

    #     output_data = output_data[:-1]

    #     rows = output_data.split('\n')
    #     # 将每行分割成字符串数组
    #     data_list = [row.split() for row in rows]
    #     # 将字符串数组转换为浮点数数组
    #     data_arr = np.array(data_list, dtype=float)
    #     poses_optimized = data_arr[:,1:]
    #     poses_optimized[:,-1] = poses_optimized[:,-1] / math.pi *180#转换到角度制度
    #     # print("estimated pose is:\n", poses_optimized)
    #     # if self.self_robot_name == "robot1":
    #     #     poses_optimized = np.array([[0,0,0],[7,7,0]])
    #     # else:
    #     #     poses_optimized = np.array([[0,0,0],[-7,-7,0]])
    #     for i in range(0,now_meeted_robot_num):
    #         now_meeted_robot_pose = poses_optimized[1+i,:]
    #         print("---------------Robot Center Optimized-----------------------\n")
    #         print(self.self_robot_name,"estimated robot pose of (x ,y, Yaw in degree)", self.meeted_robot[i],now_meeted_robot_pose)
    #         self.map_frame_pose[self.meeted_robot[i]] = list()
    #         self.map_frame_pose[self.meeted_robot[i]].append(R.from_euler('z', now_meeted_robot_pose[2], degrees=True).as_matrix()) 
    #         self.map_frame_pose[self.meeted_robot[i]].append(np.array([now_meeted_robot_pose[0],now_meeted_robot_pose[1],0]))
    
    def topo_optimize(self):
        #self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.pose,vertex.pose,pose])
        # This part should update self.map_frame_pose[vertex.robot_name];self.map_frame_pose[vertex.robot_name][0] R33;[1]t 31
        input = self.estimated_vertex_pose
        outlier_rejected_input = outlier_rejection(input)
        now_meeted_robot_num = len(self.meeted_robot) #只做单个机器人这部分
        now_meeted_robot_pose = pose_gragh_opt(outlier_rejected_input,0.2)
        for i in range(0,now_meeted_robot_num):
            print("---------------Robot Center Optimized-----------------------\n")
            print(self.self_robot_name,"estimated robot pose of (x ,y, Yaw in degree)", self.meeted_robot[i],now_meeted_robot_pose)
            self.map_frame_pose[self.meeted_robot[i]] = list()
            self.map_frame_pose[self.meeted_robot[i]].append(R.from_euler('z', now_meeted_robot_pose[2], degrees=True).as_matrix()) 
            self.map_frame_pose[self.meeted_robot[i]].append(np.array([now_meeted_robot_pose[0],now_meeted_robot_pose[1],0]))
    
    def update_relative_pose(self):
        # 定义转换关系的消息
        now_estimated_relative_pose = self.map_frame_pose[self.meeted_robot[0]]
        rela_rot = now_estimated_relative_pose[0]
        rela_trans = now_estimated_relative_pose[1]
        orientation = R.from_matrix(rela_rot).as_quat()

        transform = TransformStamped()
        transform.header.frame_id = self.self_robot_name + '/map'  # 源坐标系
        transform.child_frame_id = self.self_robot_name + '/map_origin'  # 目标坐标系
        
        # 设置转换关系的平移部分
        transform.transform.translation.x = rela_trans[0] # 在X轴上的平移量
        transform.transform.translation.y = rela_trans[1] # 在Y轴上的平移量
        transform.transform.translation.z = 0.0  # 在Z轴上的平移量
        
        # 设置转换关系的旋转部分（四元数表示）
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        transform.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(transform)


    def update_vertex_pose(self):
        now_estimated_relative_pose = self.map_frame_pose[self.meeted_robot[0]]
        rela_rot = now_estimated_relative_pose[0]
        rela_trans = now_estimated_relative_pose[1]

        for index, now_vertex in enumerate(self.received_map.vertex):
            tmp_pose = rela_rot @ np.array([now_vertex.pose[0],now_vertex.pose[1],0]) + rela_trans
            self.map.vertex[index].pose = [tmp_pose[0], tmp_pose[1],self.map.vertex[index].pose[2]]


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
            #move the robot
            if self.receive_topomap:
                self.change_goal()
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
    rospy.init_node('topo_relocalization')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    print(robot_name, robot_num)

    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    
    robot_list.remove(robot_name) #记录其他机器人id
    node = RobotNode(robot_name, robot_list)

    print("-------init robot relocalization node--------")
    #订阅自己的图像
    robot1_image1_sub = message_filters.Subscriber(robot_name+"/camera1/image_raw", Image)
    robot1_image2_sub = message_filters.Subscriber(robot_name+"/camera2/image_raw", Image)
    robot1_image3_sub = message_filters.Subscriber(robot_name+"/camera3/image_raw", Image)
    robot1_image4_sub = message_filters.Subscriber(robot_name+"/camera4/image_raw", Image)
    ts = message_filters.TimeSynchronizer([robot1_image1_sub, robot1_image2_sub, robot1_image3_sub, robot1_image4_sub], 10) #传感器信息融合
    ts.registerCallback(node.create_panoramic_callback) # 

    rospy.spin()