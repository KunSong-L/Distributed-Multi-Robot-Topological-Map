#!/usr/bin/python3.8
import math
import rospy
from std_msgs.msg import Int32, Float32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import   Point
import numpy as np
from scipy.spatial.transform import Rotation as R
from robot_function import change_frame,change_frame_multi, voronoi_region, four_point_of_a_rect
from utils.solveVRP import VRP_solver
from utils.tf_graph_manager import multi_robot_tf_manager
from utils.simple_explore import RobotNode
from robot_fht_map_creater import fht_map_creater
# from robot_fht_map_creater_NS import fht_map_creater #ablation study for node selection

import copy
from TopoMap import Support_Vertex, Vertex
import open3d as o3d
import tf
# from ransac_icp import ransac_icp
from utils.global_icp import global_icp

import os
import subprocess
from sensor_msgs.msg import Image
from robot_function import calculate_vertex_info
from TopoMap import Support_Vertex, Vertex, Edge, TopologicalMap
from utils.static_fht_map import static_fht_map
import time 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped


class multi_rendezvous_manager():
    def __init__(self, robot_num):#the maxium robot num
        self.robot_num = robot_num
        robot_list_name = []
        for rr in range(robot_num):
            robot_list_name.append("robot"+str(rr+1))
        
        #minimal class for robot exploration
        self.RobotNode_list = [RobotNode(now_robot,use_clustered_frontier=False) for now_robot in robot_list_name]
        
        #create fht map creater
        #color: frontier, main_vertex, support vertex, edge, local free space
        robot1_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF],[0x00, 0xFF, 0x00]])/255.0
        robot2_color = np.array([[0xFF, 0xA5, 0x00], [0xDC, 0x14, 0xb1], [0x16, 0x7c, 0xdf], [0x00, 0x64, 0x00], [0x40, 0xE0, 0xD0],[0xb5, 0xbc, 0x38]]) / 255.0
        robot3_color = np.array([[0x8A, 0x2B, 0xE2], [0x8B, 0x00, 0x00], [0xFF, 0xF8, 0xDC], [0x7B, 0x68, 0xEE], [0xFF, 0x45, 0x00],[0x78, 0xC2, 0xC4]]) / 255.0
        robot4_color = np.array([[0x8A, 0x2B, 0xE2], [0xCD, 0x00, 0x00], [0xFF, 0xDA, 0xB9], [0x00, 0x00, 0x80], [0xFF, 0x45, 0x00],[0x9c, 0x9c, 0x9c]]) / 255.0
        self.global_color = [robot1_color,robot2_color,robot3_color,robot4_color]
        self.fhtmap_creater_list = [fht_map_creater(now_robot,4, topo_refine="no") for now_robot in robot_list_name]
        for now_robot in range(self.robot_num):
            self.fhtmap_creater_list[now_robot].vis_color = self.global_color[now_robot]
        
        #拓扑地图融合
        self.similarity_mat_dict = {(i,j): [] for i in range(robot_num) for j in range(i,robot_num)} #(i,j): mat 
        self.similarity_th = 0.96

        if self.robot_num == 2:
            self.adj_mat_topomap = np.array([[0,1],[1,0]])
        if self.robot_num==3:
            self.adj_mat_topomap = np.array([[0,1,1],[1,0,1],[1,1,0]])
        if self.robot_num==4:
            self.adj_mat_topomap = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])

        #获得一个i，j机器人之间的观测结果  需要利用这个观察结果修改FHT-Map
        #储存结果，一个在i机器人坐标系下的pose1,一个在j机器人坐标系下的pose2
        self.estimated_vertex_pose = {(i,j): [] for i in range(robot_num) for j in range(robot_num)} #(i,j): mat 
        self.estimated_vertex_index = {(i,j): [] for i in range(robot_num) for j in range(robot_num)} #(i,j): mat 
        self.transmitted_vertex_index = {(i,j): [] for i in range(robot_num) for j in range(robot_num)} #(i,j): mat 

        self.global_fht_map = None
        

        #机器人运动控制
        self.vrp_solver = VRP_solver(None,None)
        self.reassign_voronoi = True
        self.last_connnected_graph_num = self.robot_num
        self.voronoi_graph_list = [None for i in range(self.robot_num)]

        #计算tf
        self.use_GT = True
        self.tf_graph_manager = multi_robot_tf_manager(robot_num,sub_tf_flag=False,use_GT=self.use_GT)

        #是否执行交汇
        self.perform_rende_flag = False
        self.published_a_ren_goal = False
        self.ren_goal = None

        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        self.global_frontier_publisher = rospy.Publisher('/global_frontier_points', Marker, queue_size=1)
        self.rend_time_pub = rospy.Publisher('/rend_time_pub', Float32, queue_size=1)
        self.all_time_published = False
        self.first_time_pub = False
        self.start_time = -1

        rospy.Subscriber('/need_new_goal', Int32, self.PIER_assign, queue_size=1) #PIER_assign, NBV_assign, vrp_assign
        # rospy.Subscriber("/robot1/panoramic", Image, self.fht_map_merger, queue_size=1)

        rospy.Subscriber("/robot1/panoramic", Image, self.multi_robot_rendezvous_callback, queue_size=1)
        rospy.Subscriber("/perform_rend_flag", Int32, self.perform_rend_callback, queue_size=1)
        rospy.Subscriber("/robot3/map", OccupancyGrid, self.map_grid_callback, queue_size=1) #add proper tf for merged map
        rospy.Subscriber("/merged_map", OccupancyGrid, self.mergered_map_callback, queue_size=1) #store merged map
        self.merged_map = None
        self.tf_broad = tf.TransformBroadcaster()
        self.tran_map3_merged = [0,0] #x,y
        

    def map_grid_callback(self,data):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "/robot3/map"
        t.child_frame_id = "/merged_map"
        t.transform.translation.x = data.info.origin.position.x
        t.transform.translation.y = data.info.origin.position.y
        t.transform.translation.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, 0)  # 欧拉角转四元数
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broad.sendTransformMessage(t)
        self.tran_map3_merged = [data.info.origin.position.x,data.info.origin.position.y]

    def mergered_map_callback(self,data):
        shape = (data.info.height, data.info.width)
        self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
        
        self.merged_map = np.asarray(data.data).reshape(shape)


    def perform_rend_callback(self,data):
        self.perform_rende_flag=True

    def publish_now_time(self):

        now_time_data = Float32()
        now_time_data.data = rospy.Time.now().to_sec()

        if self.start_time == -1:
            self.start_time = now_time_data.data
        now_time_data.data -= self.start_time
        self.rend_time_pub.publish(now_time_data)

    def multi_robot_rendezvous_callback(self,data):
        if not self.perform_rende_flag:
            return

        if self.published_a_ren_goal:
            goal_reached_all_num = 0
            for i in range(self.robot_num):
                if self.RobotNode_list[i].start_follow_path_flag== False:
                    goal_reached_all_num += 1
            
            if goal_reached_all_num == self.robot_num:
                if not self.all_time_published:
                    self.publish_now_time() #交汇完成时间
                    rospy.sleep(0.1)
                    total_transmmitted_vertex_num =0
                    for i in range(robot_num):
                        for j in range(robot_num):
                            total_transmmitted_vertex_num+=len(self.transmitted_vertex_index[(i,j)])
                    transmmited_scan_num = Float32()
                    transmmited_scan_num.data = total_transmmitted_vertex_num
                    self.rend_time_pub.publish(transmmited_scan_num) #传输交换了多少激光雷达数据

                    self.all_time_published=True
                print("all Goal Reached")
            return
        #use GT value
        self.tf_graph_manager.add_tf_trans(0,1,[0,0,0])
        self.tf_graph_manager.add_tf_trans(0,2,[0,0,0])
        
        
        self.published_a_ren_goal = True#进入这个函数之后机器人就不再运动了
        self.publish_now_time() #获取全部相对位姿时间
        for i in range(self.robot_num):
            self.RobotNode_list[i].change_goal(np.array(self.RobotNode_list[i].pose[0:2]),0)#停止每个机器人

        #计算机器人位置
        robot_pose_list = []
        robot_pose_frame_merged_list = [] #将机器人全部统一到3坐标系
        robot_pose_rc = []
        tran_map3_merged = copy.deepcopy(self.tran_map3_merged)
        for i in range(self.robot_num):
            robot_pose_list.append(self.RobotNode_list[i].pose[0:2]) 
            r3_p = change_frame(self.RobotNode_list[i].pose[0:2],self.tf_graph_manager.get_relative_trans(i,2))
            r3_p[0] -= tran_map3_merged[0]
            r3_p[1] -= tran_map3_merged[1]
            robot_pose_frame_merged_list.append(r3_p)
            robot_pose_rc.append([r3_p[1]//0.05, r3_p[0]//0.05])
        robot_pose_rc = np.array(robot_pose_rc,dtype=int)
        print(robot_pose_rc)
        #计算交会点
        from utils.easy_map import easy_grid_map
        now_map = easy_grid_map(copy.deepcopy(self.merged_map), [0,0],0.05)
        num_point = 5
        potential_p_rc = now_map.random_points_on_map(num_point) #生成点的位置

        total_path_length = []
        num_robot = 3

        for i in range(num_robot):
            start = (robot_pose_rc[i, 0], robot_pose_rc[i, 1])
            end = [(potential_p_rc[j, 0], potential_p_rc[j, 1]) for j in range(num_point)]
            path,length = now_map.calculate_path_between2points_r2(start, end, False)
            total_path_length.append(length)

        total_path_length = np.array(total_path_length)
        max_path_length = np.max(total_path_length,axis=0)
        min_max_length_index = np.argmin(max_path_length)
        min_max_point_rc = potential_p_rc[min_max_length_index] #交会点在merged map frame下
        min_max_point = [min_max_point_rc[1]*0.05, min_max_point_rc[0] * 0.05]
        #换到robot3坐标系下面
        min_max_p_r3 = [0,0]
        min_max_p_r3[0] = min_max_point[0] + tran_map3_merged[0]
        min_max_p_r3[1] = min_max_point[1] + tran_map3_merged[1]
        
        min_max_p_r3 = np.array(min_max_p_r3)
        print("send ", min_max_p_r3)

        #规划一条从每个机器人到交汇点的最短路径
        for i in range(self.robot_num):
            rela_pose = self.tf_graph_manager.get_relative_trans(2,i)
            path_point = change_frame_multi([min_max_p_r3], rela_pose)
            self.RobotNode_list[i].path_point = path_point
            self.RobotNode_list[i].start_follow_path_flag = True

        print("Perform Rendezvous")  
        self.publish_now_time() #开始交汇时间    



    def PIER_assign(self,data):
        if not self.first_time_pub:
            self.publish_now_time() #开始运动
            self.first_time_pub = True
        start_time = time.time()
        if self.published_a_ren_goal:
            return
        #对每个集群内部进行分区探索
        #首先获取子图
        sub_graphs = self.tf_graph_manager.obtain_sub_connected_graph()
        if len(sub_graphs) == 1:
            self.perform_rende_flag = True
        robot_index = data.data
        robot_sub_graph = None
        for now_sub_graph in sub_graphs:
            if now_sub_graph[robot_index] == True:
                robot_sub_graph = now_sub_graph
                break
        sub_graph_robot_index = np.where(robot_sub_graph == True)[0] #an index array
        now_robot_frontier = copy.deepcopy(self.RobotNode_list[robot_index].clustered_frontier)

        if len(now_robot_frontier) < 2 or len(sub_graphs) < self.last_connnected_graph_num:
            if len(sub_graph_robot_index) >1 :
                self.reassign_voronoi = True
                self.last_connnected_graph_num = len(sub_graphs)
        
        if self.reassign_voronoi:
            #重新构建voronoi图
            #将前沿点全部统一到子图内部某个坐标系下
            #共享前沿点并构建vorono图
            print(f"now robot {robot_index}")
            print("----------Begin Re Partination Space-----------")
            for now_sub_graph in sub_graphs:
                target_frame = -1
                # total_frontier = np.array([]).reshape((-1,2))
                total_pose = np.array([]).reshape((-1,2))
                in_sub_graph = []
                for index in range(self.robot_num):
                    if now_sub_graph[index] != 1:
                        continue
                    if target_frame == -1:
                        target_frame = index
                    # tmp_frontier = copy.deepcopy(self.RobotNode_list[index].clustered_frontier)
                    #统一坐标系到target_frame
                    frame_trans = self.tf_graph_manager.get_relative_trans(index,target_frame)
                    # target_frame_frontier = change_frame_multi(tmp_frontier.reshape((-1,2)),frame_trans).reshape((-1,2))
                    # total_frontier = np.vstack((total_frontier, target_frame_frontier))

                    now_robot_pose = self.RobotNode_list[index].pose[0:2]
                    target_frame_pose = change_frame(now_robot_pose,frame_trans) 
                    total_pose = np.vstack((total_pose, target_frame_pose))
                    in_sub_graph.append(index)

                # 前沿点一起管理，然后分配算法改一下
                for index in range(self.robot_num):
                    if now_sub_graph[index] != 1:
                        continue
                    frame_trans = self.tf_graph_manager.get_relative_trans(target_frame, index)
                    # robot_frame_frontier = change_frame_multi(total_frontier, frame_trans)
                    # self.RobotNode_list[index].clustered_frontier = robot_frame_frontier
                    self.voronoi_graph_list[index] = voronoi_region(change_frame_multi(total_pose, frame_trans), in_sub_graph) #自己维护一个voronoi diagram
            self.reassign_voronoi = False
            print("----------End Re Partination Space-----------")
        
        #1. 收集所有在一个集群内的frontier
        total_frontier = np.array([]).reshape((-1,2))
        target_frame = sub_graph_robot_index[0]
        for now_robot_index in sub_graph_robot_index:
            tmp_frontier = copy.deepcopy(self.RobotNode_list[now_robot_index].clustered_frontier)
            #统一坐标系到target_frame
            frame_trans = self.tf_graph_manager.get_relative_trans(now_robot_index,target_frame)
            target_frame_frontier = change_frame_multi(tmp_frontier.reshape((-1,2)),frame_trans).reshape((-1,2))
            total_frontier = np.vstack((total_frontier, target_frame_frontier))
        end_time = time.time()
        # print("time for collecting frontier", end_time-start_time)
        #2. 判断fontier是否被探索
        frontier_number = len(total_frontier)
        not_expored_index = [True for i in range(frontier_number)]
      
        for now_robot_index in sub_graph_robot_index:
            frame_trans = self.tf_graph_manager.get_relative_trans(target_frame, now_robot_index)
            robot_frame_frontier = change_frame_multi(total_frontier, frame_trans) #robot frame

            for frontier_index, now_frontier in enumerate(robot_frame_frontier):
                explored_frontier_flag  = self.RobotNode_list[now_robot_index].is_explored_frontier(now_frontier)
                if explored_frontier_flag:
                    not_expored_index[frontier_index] = False
        
        end_time = time.time()
        # print("time for detect is frontier explored", end_time-start_time)
        real_total_frontier = total_frontier[not_expored_index,:]
        if len(real_total_frontier) == 0:
            # print("no frontier in this group")
            return

        #3. 进行分配
        #将前沿点转到该机器人坐标系下
        frame_trans = self.tf_graph_manager.get_relative_trans(target_frame, robot_index)
        real_frontier_RF = change_frame_multi(real_total_frontier, frame_trans) #robot frame
        
        voronoi_partition = self.voronoi_graph_list[robot_index].find_region(real_frontier_RF) #
        partition_robot_index = np.array(self.voronoi_graph_list[robot_index].keys_of_region(voronoi_partition))
        # print("partition_robot_index",partition_robot_index)
        in_partition_fontier = real_frontier_RF[partition_robot_index == robot_index]
        #可能存在没有任何点的情况

        if len(in_partition_fontier) == 0:
            self.no_frontier_assign(robot_index, real_frontier_RF, sub_graph_robot_index)
            return
        
        #Local NBV Assign
        robot_pose = copy.deepcopy(self.RobotNode_list[robot_index].pose)
        frontier_scores = self.frontier_ulitity_function(robot_pose,in_partition_fontier)
        min_index = np.argmin(frontier_scores)
        choose_frontier = copy.deepcopy(in_partition_fontier[min_index])
        if not self.perform_rende_flag:
            self.RobotNode_list[robot_index].change_goal(choose_frontier,0)
        end_time = time.time()
        # print("time for PIER Assign", end_time-start_time)

    def frontier_ulitity_function(self,robot_pose,frontier_poses):
        dis_frontier_poses = np.sqrt(np.sum(np.square(frontier_poses - robot_pose[0:2]), axis=1))
        dis_cost = np.abs(dis_frontier_poses)

        angle_frontier_poses = np.arctan2(frontier_poses[:, 1] - robot_pose[1], frontier_poses[:, 0] - robot_pose[0]) - robot_pose[2] / 180 * np.pi
        angle_frontier_poses = np.arctan2(np.sin(angle_frontier_poses), np.cos(angle_frontier_poses)) # turn to -pi~pi
        angle_cost = np.abs(angle_frontier_poses)
        dis_epos = 2 #线速度0.5
        angle_epos = 5 #角速度0.2

        frontier_scores = dis_epos * dis_cost + angle_epos * angle_cost
        target_score = 10
        return np.abs(frontier_scores - target_score)
      

    def no_frontier_assign(self,robot_index,total_frontier,sub_graph_robot_index):
        # robot_index: index of this robot
        # total_frontier: total not explored frontier in this sub graph
        # sub_graph_robot_index: somethis like [0,1,2], which indicates this index of robot in this goup

        #计算距离最远的
        total_ultility = []
        for now_robot_index in sub_graph_robot_index:
            if now_robot_index==robot_index:
                continue
            robot_pose = self.RobotNode_list[now_robot_index].pose
            frame_trans = self.tf_graph_manager.get_relative_trans(robot_index, now_robot_index) 
            robot_frame_frontier = change_frame_multi(total_frontier, frame_trans).reshape((-1,2)) #robot frame
            robot_ultility = self.frontier_ulitity_function(robot_pose, robot_frame_frontier)
            total_ultility.append(robot_ultility)
        
        max_ulti = np.max(np.array(total_ultility),axis = 0)

        max_fontier_index = np.argmax(max_ulti)
        choose_frontier = copy.deepcopy(total_frontier[max_fontier_index])
        self.RobotNode_list[robot_index].change_goal(choose_frontier,0)
       

    def NBV_assign(self,data):
        if self.published_a_ren_goal:
            return
        #对每个集群内部进行VRP分配前沿点
        #首先获取子图
        if not self.first_time_pub:
            self.publish_now_time() #开始运动
            self.first_time_pub = True
        sub_graphs = self.tf_graph_manager.obtain_sub_connected_graph()
        if len(sub_graphs) == 1:
            self.perform_rende_flag = True
        robot_index = data.data
        #判断当前机器人在哪个子图下
        for now_sub_graph in sub_graphs:
            if now_sub_graph[robot_index] == True:
                break
        
        total_fontier = copy.deepcopy(self.RobotNode_list[robot_index].total_frontier)
        robot_pose = copy.deepcopy(self.RobotNode_list[robot_index].pose)

        frontier_poses = total_fontier
        dis_frontier_poses = np.sqrt(np.sum(np.square(frontier_poses - robot_pose[0:2]), axis=1))
        dis_cost = np.abs(dis_frontier_poses - 2)

        angle_frontier_poses = np.arctan2(frontier_poses[:, 1] - robot_pose[1], frontier_poses[:, 0] - robot_pose[0]) - robot_pose[2] / 180 * np.pi
        angle_frontier_poses = np.arctan2(np.sin(angle_frontier_poses), np.cos(angle_frontier_poses)) # turn to -pi~pi
        angle_cost = np.abs(angle_frontier_poses)
        
        # calculate frontier information
        vertex_info = np.array(calculate_vertex_info(frontier_poses))
        dis_epos = 1
        angle_epos = 2
        frontier_scores = (1 + np.exp(-vertex_info))*(dis_epos * dis_cost + angle_epos * angle_cost)
        max_index = np.argmin(frontier_scores)

        choose_frontier = copy.deepcopy(frontier_poses[max_index])
        try:#可能这个前沿点已经在其他地方被删除过了
            np.delete(self.RobotNode_list[robot_index].total_frontier, max_index, axis=0)
        except:
            pass
        
        #修改前沿点
        self.RobotNode_list[robot_index].change_goal(choose_frontier,0)
        
    def vrp_assign(self,data):
        if self.published_a_ren_goal:
            return
        if not self.first_time_pub:
            self.publish_now_time() #开始运动
            self.first_time_pub = True
        #对每个集群内部进行VRP分配前沿点
        #首先获取子图
        sub_graphs = self.tf_graph_manager.obtain_sub_connected_graph()
        if len(sub_graphs) == 1:
            self.perform_rende_flag = True
        robot_index = data.data
        #判断当前机器人在哪个子图下
        for now_sub_graph in sub_graphs:
            if now_sub_graph[robot_index] == True:
                break
        #将前沿点全部统一到子图内部某个坐标系下
        start = time.time()
        target_frame = -1
        total_frontier = np.array([]).reshape((-1,2))
        robot_pose = np.array([]).reshape((-1,3))#机器人是带朝向的
        for index in range(self.robot_num):
            if now_sub_graph[index] != 1:
                continue
            if target_frame == -1:
                target_frame = index
            now_robot_frontier = copy.deepcopy(self.RobotNode_list[index].clustered_frontier)
            #统一坐标系到target_frame
            frame_trans = self.tf_graph_manager.get_relative_trans(index,target_frame)
            target_frame_frontier = change_frame_multi(now_robot_frontier,frame_trans).reshape((-1,2))
            total_frontier = np.vstack((total_frontier, target_frame_frontier))

            #把机器人的位置也改变一下坐标系
            now_robot_pose = self.RobotNode_list[index].pose
            target_frame_pose = change_frame(now_robot_pose,frame_trans) 
            robot_pose = np.vstack((robot_pose, target_frame_pose))
        if len(total_frontier)==0:
            print("Exploration Finished!")
            return
        #需要判断前沿点是否被其他机器人探索过
        not_expored_index = [True for i in range(len(total_frontier))]
        for index in range(self.robot_num):
            if now_sub_graph[index] != 1:
                continue
            #把前沿点都转换到index frame下
            frame_trans = self.tf_graph_manager.get_relative_trans(target_frame,index)
            robot_frame_frontier = change_frame_multi(total_frontier,frame_trans)
            #判断改前沿点是否还是一个前沿
            
            for frontier_index, now_frontier in enumerate(robot_frame_frontier):
                explored_frontier_flag  = self.RobotNode_list[index].is_explored_frontier(now_frontier)
                if explored_frontier_flag:
                    not_expored_index[frontier_index] = False
        #筛选前沿点
        real_total_frontier = total_frontier[not_expored_index,:]
        if len(real_total_frontier) == 0:
            print("Find no frontier")
            return
        
        #求解VRP问题
        # self.vrp_solver.robot_pose = robot_pose[:,0:2] #只采用距离代价，先不考虑旋转
        # self.vrp_solver.points = real_total_frontier #仅分配还没探索过的
        # result_path,path_length = self.vrp_solver.solveVRP()
        #看起来存在一点问题，不一定实在robot index = 0的机器人上进行拓扑路径规划
        C_robot_to_frontier, C_frontier_to_frontier = self.create_costmat(robot_pose, real_total_frontier, 0) #利用拓扑图上的距离求解VRP问题
        result_path,path_length = self.vrp_solver.solveVRP(C_robot_to_frontier, C_frontier_to_frontier)
        #可视化前沿点
        self.visulize_frontier("robot"+str(target_frame+1),total_frontier[not_expored_index,:])
        end =  time.time()
        print(f" time for assign robot {robot_index+1}, {end - start}")
        #分配任务
        group_robot_index = -1
        for now_robot_index in range(self.robot_num):
            if now_sub_graph[now_robot_index]: #计算当前下标为now_robot_index的机器人是当前集群中的第几个机器人
                group_robot_index+=1
            else:
                continue
            
            if now_robot_index == robot_index: #当前机器人为需要发布前沿点的机器人
                robot_path = result_path[group_robot_index+1]
                if len(robot_path)==0:
                    continue
                target_frontier = real_total_frontier[robot_path[0] - 1] #target_frame下
                #修改前沿点坐标系并发送前沿点
                robot_frame_frontier = change_frame(target_frontier,self.tf_graph_manager.get_relative_trans(target_frame,now_robot_index))
                self.RobotNode_list[now_robot_index].change_goal(np.array(robot_frame_frontier),0)

    
    def create_costmat(self,robot_pose,frontier_pose,robot_index):
        n_robot = len(robot_pose)
        n_point = len(frontier_pose)
        C_robot_to_frontier = np.zeros((n_robot,n_point))
        C_frontier_to_frontier = np.zeros((n_point,n_point))
        for i in range(n_robot):
            for j in range(n_point):
                C_robot_to_frontier[i,j],path_point = self.fhtmap_creater_list[robot_index].topo_path_planning(robot_pose[i],frontier_pose[j])

        for i in range(n_point):
            for j in range(n_point):
                C_frontier_to_frontier[i,j],path_point = self.fhtmap_creater_list[robot_index].topo_path_planning(frontier_pose[i],frontier_pose[j])
        
        # #处理最近的前沿点
        beta = 0 #对于最近的点施加一个很小的代价
        for now_robot_to_frontier in C_robot_to_frontier:
            min_index = np.argmin(now_robot_to_frontier)
            now_robot_to_frontier[min_index] += beta

        return C_robot_to_frontier, C_frontier_to_frontier


    def visulize_frontier(self,robot_name,frontiers):
        # ----------visualize clustered frontier------------
        frontier_marker = Marker()
        now = rospy.Time.now()
        frontier_marker.header.frame_id = robot_name + "/map"
        frontier_marker.header.stamp = now
        frontier_marker.ns = "total_frontier"
        frontier_marker.type = Marker.POINTS
        frontier_marker.action = Marker.ADD
        frontier_marker.pose.orientation.w = 1.0
        frontier_marker.scale.x = 0.3
        frontier_marker.scale.y = 0.3
        frontier_marker.color.r = self.vis_color[0][0]
        frontier_marker.color.g = self.vis_color[0][1]
        frontier_marker.color.b = self.vis_color[0][2]
        frontier_marker.color.a = 0.7
        for frontier in frontiers:
            point_msg = Point()
            point_msg.x = frontier[0]
            point_msg.y = frontier[1]
            point_msg.z = 0.3
            frontier_marker.points.append(point_msg)
        self.global_frontier_publisher.publish(frontier_marker)
        # --------------finish visualize frontier---------------


            
if __name__ == '__main__':
    rospy.init_node('multi_robot_explore')
    robot_num = rospy.get_param("~robot_num")

    real_robot_explore_manager = multi_rendezvous_manager(robot_num)

    rospy.spin()

    