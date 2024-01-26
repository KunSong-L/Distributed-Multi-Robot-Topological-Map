#!/usr/bin/python3.8
#收集信息构建FHT-Map，机器人不运动
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

class fht_map_creater:
    def __init__(self, robot_name,main_node_density=25):#输入当前机器人，其他机器人的id list
        rospack = rospkg.RosPack()
        self.self_robot_name = robot_name
        self.robot_index = int(robot_name[-1])-1
        path = rospack.get_path('multi_fht_map')

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
        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        #finish init NN

        # init image sub
        image1_sub = message_filters.Subscriber(robot_name+"/camera1/image_raw", Image)
        image2_sub = message_filters.Subscriber(robot_name+"/camera2/image_raw", Image)
        image3_sub = message_filters.Subscriber(robot_name+"/camera3/image_raw", Image)
        image4_sub = message_filters.Subscriber(robot_name+"/camera4/image_raw", Image)
        ts = message_filters.TimeSynchronizer([image1_sub, image2_sub, image3_sub, image4_sub], 10) #传感器信息融合
        ts.registerCallback(self.create_panoramic_callback) # 

        #color: frontier, main_vertex, support vertex, edge, local free space
        robot1_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF],[0x00, 0xFF, 0x00]])/255.0
        robot2_color = np.array([[0xFF, 0xA5, 0x00], [0xDC, 0x14, 0xb1], [0x16, 0x7c, 0xdf], [0x00, 0x64, 0x00], [0x40, 0xE0, 0xD0],[0xb5, 0xbc, 0x38]]) / 255.0
        robot3_color = np.array([[0x8A, 0x2B, 0xE2], [0x8B, 0x00, 0x00], [0xFF, 0xF8, 0xDC], [0x7B, 0x68, 0xEE], [0xFF, 0x45, 0x00],[0xF0, 0xF8, 0xFF]]) / 255.0
        self.vis_color = [robot1_color,robot2_color,robot3_color]
        
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))

        #robot data
        self.pose = [0,0,0] # x y yaw angle in radian
        self.max_v = 0.5
        self.max_w = 0.2

        self.laserProjection = LaserProjection()
        self.cv_bridge = CvBridge()
        #topomap
        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.last_free_vertex = None #last free support vertex
        self.last_vertex_id = -1
        self.current_node = None
        self.adj_list = dict()#拓扑地图到邻接矩阵
        self.potential_main_vertex = list()
        self.map_origin=None
        self.main_vertex_dens = main_node_density

        #for relative pose estimation
        self.current_feature = [] #feature, local laser scan, pose


        # get tf
        self.tf_listener = tf.TransformListener()
        self.laser_scan_cos_sin = None
        self.laser_scan_init = False
        self.local_laserscan = None
        self.local_laserscan_angle = None

        #publisher and subscriber
        self.marker_pub = rospy.Publisher(robot_name+"/visualization/marker", MarkerArray, queue_size=1)
        self.edge_pub = rospy.Publisher(robot_name+"/visualization/edge", MarkerArray, queue_size=1)
        self.panoramic_view_pub = rospy.Publisher(robot_name+"/panoramic", Image, queue_size=1)
        self.topomap_pub = rospy.Publisher(robot_name+"/topomap", TopoMapMsg, queue_size=1)
        self.start_pub = rospy.Publisher("/start_exp", String, queue_size=1) #发一个start
        self.vertex_free_space_pub = rospy.Publisher(robot_name+'/vertex_free_space', MarkerArray, queue_size=1)
        self.find_better_path_pub = rospy.Publisher(robot_name+'/find_better_path', String, queue_size=100)
        rospy.Subscriber(robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(robot_name+"/scan", LaserScan, self.laserscan_callback, queue_size=1)
        rospy.Subscriber(robot_name+'/find_better_path', String, self.find_better_path_callback, queue_size=100)
        rospy.Subscriber(robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)
        print("Finish Init FHT-Map of ", robot_name)

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
        if len(self.local_laserscan_angle) == 0:
            return 0
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
            cv2.imwrite(debug_path + "/2.png",self.panoramic_view)
        image_message = self.cv_bridge.cv2_to_imgmsg(self.panoramic_view, encoding="rgb8")
        image_message.header.stamp = rospy.Time.now()  
        image_message.header.frame_id = self.self_robot_name+"/odom"
        self.panoramic_view_pub.publish(image_message)


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


    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        try:
            self.tf_listener.waitForTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
            tf_transform, rotation = self.tf_listener.lookupTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow)
            self.pose[0] = tf_transform[0]
            self.pose[1] = tf_transform[1]
            self.pose[2] = R.from_quat(rotation).as_euler('xyz', degrees=False)[2]

        except:
            pass
    
    def map_grid_callback(self, data):
        
        #generate grid map and global grid map
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)

        self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
        
        self.global_map_tmp = np.asarray(data.data).reshape(shape)
        self.global_map_tmp[np.where(self.global_map_tmp==-1)] = 255
        self.global_map = copy.deepcopy(self.global_map_tmp)
        #获取当前一个小范围的grid map

    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        self.update_robot_pose() #update robot pose
        if self.map_origin is None:
            return
        current_pose = copy.deepcopy(self.pose)
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        now_feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
        #创建当前所有特征
        self.current_feature = [copy.deepcopy(self.local_laserscan), copy.deepcopy(now_feature), copy.deepcopy(current_pose)]
        
        create_a_vertex_flag = self.create_a_vertex(panoramic_view,now_feature) # whether create a vertex
        if create_a_vertex_flag: # create a vertex
            if create_a_vertex_flag == 1:#create a main vertex
                omega_ch = np.array([1,2]) 
                ch_list = [] #C for localization ability; H for heat value
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
                vertex = Support_Vertex(self.self_robot_name, id=-1, pose=current_pose)
                self.last_vertex_id, self.current_node = self.map.add(vertex)
            # add rect to vertex
            if self.last_vertex_id > 0:
                self.create_local_free_space_for_single_vertex(-2)

            #create edge
            self.create_edge()
           
            self.visulize_vertex()

            # 不发布拓扑地图，直接在外层类读取
            # topomap_message = TopomapToMessage(self.map)    
            # self.topomap_pub.publish(topomap_message) # publish topomap important!      

            if create_a_vertex_flag ==1 or create_a_vertex_flag ==2: 
                refine_topo_map_msg = String()
                refine_topo_map_msg.data = "Start_find_path!"
                self.find_better_path_pub.publish(refine_topo_map_msg) #find a better path
    
    def create_local_free_space_for_single_vertex(self,index):
        self.map.vertex[index].local_free_space_rect  = find_local_max_rect(self.global_map, self.map.vertex[index].pose[0:2], self.map_origin, self.map_resolution)
    
    def add_a_support_node(self, vertex_pose):
        #增加一个支撑节点并连接边
        # print("add vertex for ",self.self_robot_name," vertex pose = ",vertex_pose)
        vertex = Support_Vertex(self.self_robot_name, id=-1, pose=copy.deepcopy(vertex_pose))
        self.last_vertex_id, self.current_node = self.map.add(vertex)
        self.visulize_vertex()
        self.create_local_free_space_for_single_vertex(-2)
        self.create_edge()

    def create_a_vertex(self,panoramic_view,now_feature):
        #return 1 for uncertainty value reach th; 2 for not a free space line
        #and 0 for don't creat a vertex
        #important parameter
        if self.local_laserscan_angle is None:
            return
        feature_simliar_th = 0.94
        main_vertex_dens = self.main_vertex_dens #main_vertex_dens^0.5 is the average distance of a vertex, 4 is good; sigma_c in paper
        global_vertex_dens = 2 # create a support vertex large than 2 meter
        # check the heat map value of this point
        local_laserscan_angle = copy.deepcopy(self.local_laserscan_angle)
        heat_map_value = self.heat_value_eval()
        
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
                vertex = Vertex(self.self_robot_name, id=-1, pose=copy.deepcopy(self.pose), descriptor=copy.deepcopy(now_feature), local_image=gray_local_img, local_laserscan_angle=local_laserscan_angle)
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
                free_line_flag = self.expanded_free_space_line(last_vertex_pose_pixel, now_robot_pose, 3)
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


    def topo_path_planning(self,point1,point2,consider_ori = True):
        #输入两个点: point1/2: 格式可能为[x,y,yaw]或者[x,y]
        #输出两个点之间规划的一条轨迹和轨迹的时间代价
        #在这个函数中，考虑了机器人转向的代价
        if len(self.map.vertex) == 0:
            return np.linalg.norm(point1[0:2] - point2[0:2]),[point1,point2] #TODO
        
        if np.linalg.norm(np.array(point1[0:2]) - np.array(point2[0:2]))<1:
            return self.path_distance_cost(point1,point2,[]),[point1,point2]

        point1_pixel = (np.array(point1[0:2])- np.array(self.map_origin))/self.map_resolution
        point2_pixel = (np.array(point2[0:2])- np.array(self.map_origin))/self.map_resolution
        if self.free_space_line(point1_pixel,point2_pixel):
            return self.path_distance_cost(point1,point2,[]),[point1,point2]

        start_in_vertex_index = self.dual_vertex_of_a_point(point1)
        target_in_vertex_index = self.dual_vertex_of_a_point(point2)
        
        self.adj_list = dict()
        self.edge_to_adj_list()
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
        





if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('topological_map')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")

    node = fht_map_creater(robot_name)   
    print("node init done")
    rospy.spin()