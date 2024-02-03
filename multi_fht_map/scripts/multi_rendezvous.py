#!/usr/bin/python3.8
import math
import rospy
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import   Point
import numpy as np
from scipy.spatial.transform import Rotation as R
from robot_function import change_frame,change_frame_multi
from utils.solveVRP import VRP_solver
from utils.tf_graph_manager import multi_robot_tf_manager
from utils.simple_explore import RobotNode
from robot_fht_map_creater import fht_map_creater
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
save_result = False

class multi_rendezvous_manager():
    def __init__(self, robot_num):#the maxium robot num
        self.robot_num = robot_num
        robot_list_name = []
        for rr in range(robot_num):
            robot_list_name.append("robot"+str(rr+1))
        
        #minimal class for robot exploration
        self.RobotNode_list = [RobotNode(now_robot) for now_robot in robot_list_name]
        
        #create fht map creater
        self.fhtmap_creater_list = [fht_map_creater(now_robot,4) for now_robot in robot_list_name]
        

        #拓扑地图融合
        self.mergered_vertex_index = [0 for i in range(robot_num)]
        self.similarity_mat_dict = {(i,j): [] for i in range(robot_num) for j in range(i,robot_num)} #(i,j): mat 
        self.similarity_th = 0.97
        self.tf_broadcaster = tf.TransformBroadcaster() #发布计算得到的相对位姿变换
        if self.robot_num == 2:
            self.adj_mat_topomap = np.array([[0,1],[1,0]])
        if self.robot_num==3:
            self.adj_mat_topomap = np.array([[0,1,1],[1,0,1],[1,1,0]])

        #获得一个i，j机器人之间的观测结果  需要利用这个观察结果修改FHT-Map
        #储存结果，一个在i机器人坐标系下的pose1,一个在j机器人坐标系下的pose2
        self.estimated_vertex_pose = {(i,j): [] for i in range(robot_num) for j in range(robot_num)} #(i,j): mat 
        self.global_fht_map = None
        
        #交汇地点选取
        self.mergered_topological_map = TopologicalMap(robot_name="robot1", threshold=0.97)

        #机器人运动控制
        self.vrp_solver = VRP_solver(None,None)

        #计算tf
        self.tf_graph_manager = multi_robot_tf_manager(robot_num)

        #是否执行交汇
        self.perform_rende_flag = False
        self.published_a_ren_goal = False
        self.ren_goal = None

        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        self.global_frontier_publisher = rospy.Publisher('/global_frontier_points', Marker, queue_size=1)
        rospy.Subscriber('/need_new_goal', Int32, self.NBV_assign, queue_size=100) #NBV_assign, vrp_assign
        rospy.Subscriber("/robot1/panoramic", Image, self.fht_map_merger, queue_size=1)
        rospy.Subscriber("/robot1/panoramic", Image, self.multi_robot_rendezvous_callback, queue_size=1)
       

    def multi_robot_rendezvous_callback(self,data):
        if not self.perform_rende_flag:
            return

        if self.published_a_ren_goal:
            self.global_fht_map.visulize_vertex()
            for i in range(self.robot_num):
                self.RobotNode_list[i].change_goal(np.array(self.ren_goal[i])[0:2],0)
            return
        
        self.published_a_ren_goal = True#进入这个函数之后机器人就不再运动了

        robot_pose_list = []
        robot_pose_frame_1_list = []
        for i in range(self.robot_num):
            robot_pose_list.append(self.RobotNode_list[i].pose[0:2]) #pose考虑旋转代价，如果不考虑输入前两个维度即可
            robot_pose_frame_1_list.append(change_frame(self.RobotNode_list[i].pose[0:2],self.tf_graph_manager.get_relative_trans(i,0)))
        
        self.merge_topomap() #融合拓扑地图

        #对所有节点计算最短路径
        each_robot_all_vertex_dis = []

        for i in range(self.robot_num):
            now_robot_pose = robot_pose_frame_1_list[i]
            now_paths = self.global_fht_map.from_start_point_to_every_vertex(now_robot_pose)
            each_robot_all_vertex_dis.append(now_paths)
        
        each_robot_all_vertex_dis = np.array(each_robot_all_vertex_dis)


        max_vertex_dis = np.max(each_robot_all_vertex_dis,axis=0)
        #对上下界进行估计
        max_freespace_dis = [] #矩形顶点距离中心的最大距离
        for now_vertex in self.global_fht_map.map.vertex:
            now_free_space = now_vertex.local_free_space_rect
            x1,y1,x2,y2 = now_free_space

            tmp = now_vertex.rotation.T @ np.array([x2-x1,y2-y1,0]) #矩形的x y方向长度
            direction1 = now_vertex.rotation @ np.array([1,0,0])
            direction2 = now_vertex.rotation @ np.array([0,1,0])

            x3y3 = direction1 * tmp[0]
            x3 = x3y3[0]
            y3 = x3y3[1]
            x4y4 = direction2 * tmp[0]
            x4 = x4y4[0]
            y4 = x4y4[1]

            four_point = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])

            vertex_pose = np.array(now_vertex.pose[0:2])
            four_dis = np.linalg.norm(four_point - vertex_pose,axis=1)
            max_dis = np.max(four_dis)
            max_freespace_dis.append(max_dis)
        
        max_freespace_dis = np.array(max_freespace_dis)

        upper_bound = max_vertex_dis + max_freespace_dis
        lower_bound = max_vertex_dis - max_freespace_dis

        min_upper_bound = np.min(upper_bound)
        potential_opt_point_index = lower_bound < min_upper_bound

        #接下去对potential_opt_point_index中的所有点取最短路径
        print("potential_opt_point_index: ", potential_opt_point_index)
        #对潜在的所有local free space遍历求解最短路径
        delta = 0.2 #local free space插值的密度
        total_rend_point = []
        total_rend_dis = []
        for i in range(len(self.global_fht_map.map.vertex)):
            if not potential_opt_point_index[i]:
                continue

            now_vertex = self.global_fht_map.map.vertex[i]
            now_free_space = now_vertex.local_free_space_rect
            x1,y1,x2,y2 = now_free_space

            #插值
            tmp = now_vertex.rotation.T @ np.array([x2-x1,y2-y1,0]) #矩形的x y方向长度
            freespace_wh = tmp[0:2].reshape(2) #width, height

            x_grid, y_grid = np.meshgrid(np.arange(0, freespace_wh[0], delta), np.arange(0, freespace_wh[1], delta))
            points = np.vstack((x_grid.flatten(), y_grid.flatten()))

            real_points = now_vertex.rotation[0:2,0:2] @ points + np.array([[x1],[y1]]) #旋转回来, 2*n
            real_points = real_points.T

            #对于任意个点，计算最长路径
            if len(real_points) == 0:
                continue
            point_dis = []
            for now_point in real_points:
                #计算adj node
                adj_node_index = self.global_fht_map.dual_vertex_of_a_point(now_point)

                #对于每个机器人计算最短路径
                robot_dis = []
                for robot_index in range(self.robot_num):
                    #最短距离就是到拓扑节点距离加上节点到这个点距离
                    node_dis = []
                    for now_node_index in adj_node_index:
                        node_pose = np.array(self.global_fht_map.map.vertex[now_node_index].pose[0:2])
                        to_node_dis = each_robot_all_vertex_dis[robot_index][now_node_index]
                        total_dis = to_node_dis + np.linalg.norm(node_pose - now_point)
                        node_dis.append(total_dis)

                    min_dis = np.min(node_dis)
                    robot_dis.append(min_dis)
                point_dis.append(np.max(robot_dis))
            #找到最短的点
            min_index = np.argmin(point_dis)
            total_rend_point.append(real_points[min_index])
            total_rend_dis.append(point_dis[min_index])
        
        #找到全局最小的
        global_min_index = np.argmin(total_rend_dis)
        target_in_robot1_frame = total_rend_point[global_min_index]
        
        #找最小下标
        # max_index = np.argmin(max_vertex_dis)
        # vertex_pose = self.global_fht_map.map.vertex[max_index].pose
        # #直接发送
        # target_in_robot1_frame = vertex_pose
        target_list = []
        target_list.append(target_in_robot1_frame)
        for i in range(1,self.robot_num):
            rela_pose = self.tf_graph_manager.get_relative_trans(0,i)
            target_i_frame = change_frame(target_in_robot1_frame,rela_pose)
            target_list.append(target_i_frame)
        
        self.ren_goal = target_list
        for i in range(self.robot_num):
            self.RobotNode_list[i].change_goal(np.array(self.ren_goal[i])[0:2],0)
            print(f"Publish a goal to robot {i+1}: {self.ren_goal[i]}")
        print("Perform Rendezvous")


    def merge_topomap(self):
        #融合所有的拓扑地图
        #首先处理一下每个机器人的拓扑地图，对于发生观测的地方创建一个节点
        print("-----------Merging Map----------------------")
        added_vertex_id = [] #((robot1,id1),(robot2,id2))
        for robot1_index in range(self.robot_num ):
            for robot2_index in range(self.robot_num):
                now_est_list = self.estimated_vertex_pose[(robot1_index,robot2_index)]
                if len(now_est_list) != 0:
                    for now_est in now_est_list:
                        robot1_pose = now_est[0]
                        robot2_pose = now_est[1]
                        self.fhtmap_creater_list[robot1_index].add_a_support_node(robot1_pose) #TODO
                        self.fhtmap_creater_list[robot2_index].add_a_support_node(robot2_pose) #只要加了这两行代码就会有问题

                        robot_name1 = self.fhtmap_creater_list[robot1_index].self_robot_name
                        robot_name2 = self.fhtmap_creater_list[robot2_index].self_robot_name
                        robot_id1 = self.fhtmap_creater_list[robot1_index].map.vertex_id
                        robot_id2 = self.fhtmap_creater_list[robot2_index].map.vertex_id
                        added_vertex_id.append(((robot_name1,robot_id1),(robot_name2,robot_id2)))
        
        #全部融合到机器人1坐标系下
        global_vertex_list = []
        global_edge_list = []
        remap_dict = dict() #(robot_name, id) : new_id
        vertex_id_index = 0
        for i in range(self.robot_num):
            now_topomap = copy.deepcopy(self.fhtmap_creater_list[i].map)
            #获取相对位姿变换
            relative_pose = self.tf_graph_manager.get_relative_trans(0,i) #获取从0到机器人i的相对位姿变换
            print(f"get relative pose from 0 to {i} {relative_pose}")
            relative_rot = R.from_euler('z', relative_pose[2], degrees=False).as_matrix()
            relative_trans = np.array([relative_pose[0],relative_pose[1],0])
            now_topomap.change_topomap_frame([relative_rot,relative_trans])
            #添加入节点
            for now_vertex in now_topomap.vertex:
                tmp = copy.deepcopy(now_vertex)
                tmp.id = vertex_id_index
                tmp.rotation = copy.deepcopy(relative_rot) #加入旋转
                global_vertex_list.append(tmp)
                vertex_id_index+=1
                remap_dict[(now_vertex.robot_name, now_vertex.id)] = tmp.id
            #添加入边
            for now_edge in now_topomap.edge:
                # edge格式：[[last_robot_name, last_robot_id], [now_robot_name, now_vertex_id]]
                new_edge = copy.deepcopy(now_edge)
                new_edge.link[0][1] = remap_dict[(new_edge.link[0][0], new_edge.link[0][1])]
                new_edge.link[1][1] = remap_dict[(new_edge.link[1][0], new_edge.link[1][1])]
                global_edge_list.append(new_edge)
        
        #把相互观测的边也加进去
        for now_est_edge in added_vertex_id:
            link = [[now_est_edge[0][0], remap_dict[(now_est_edge[0][0], now_est_edge[0][1])]], [now_est_edge[1][0], remap_dict[(now_est_edge[1][0], now_est_edge[1][1])]]]
            edge_id = len(global_edge_list)
            new_edge = Edge(edge_id,link)
            # new_edge = copy.deepcopy(now_est_edge) #TODO
            # new_edge.link[0][1] = remap_dict[(new_edge[0][0], new_edge[0][1])]
            # new_edge.link[1][1] = remap_dict[(new_edge[1][0], new_edge[1][1])]
            global_edge_list.append(new_edge)

        #获取一个更新后的拓扑地图
        self.mergered_topological_map.vertex = global_vertex_list
        self.mergered_topological_map.edge = global_edge_list
        self.global_fht_map = static_fht_map("robot0", self.mergered_topological_map)
        self.global_fht_map.visulize_vertex() #可视化


    def NBV_assign(self,data):
        if self.published_a_ren_goal:
            return
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
            target_frame_frontier = change_frame_multi(now_robot_frontier,frame_trans)
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
                explored_frontier_flag  = self.RobotNode_list[index].is_explored_frontier(now_frontier,True)
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

        C_robot_to_frontier, C_frontier_to_frontier = self.create_costmat(robot_pose, real_total_frontier, 0) #利用拓扑图上的距离求解VRP问题
        result_path,path_length = self.vrp_solver.solveVRP(C_robot_to_frontier, C_frontier_to_frontier)
        #可视化前沿点
        self.visulize_frontier("robot"+str(target_frame+1),total_frontier[not_expored_index,:])
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
                self.RobotNode_list[now_robot_index].change_goal(np.array(robot_frame_frontier))

    
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
        beta = -50 #对于最近的点施加一个很小的代价
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

    def fht_map_merger(self,data):
        if self.published_a_ren_goal:
            return
        obtain_new_est = False
        for robot1_index in range(self.robot_num): #对robot1 进行分析
            now_features = copy.deepcopy(self.fhtmap_creater_list[robot1_index].current_feature)
            if len(now_features) == 0:
                continue

            now_local_laser_scan_angle = now_features[0]
            now_descriptor = now_features[1]
            now_pose = now_features[2]
            virtual_vertex = Vertex(self.fhtmap_creater_list[robot1_index].self_robot_name, id=-1, pose=now_pose, descriptor=now_descriptor, local_image=None, local_laserscan_angle=now_local_laser_scan_angle)
            
            for robot2_index in range(self.robot_num): #与robot2 的地图进行匹配
                if self.adj_mat_topomap[robot1_index,robot2_index]==0: #只匹配互相传输地图的
                    continue
                
                fht_map2 = self.fhtmap_creater_list[robot2_index].map
                
                for index2, now_vertex_2 in enumerate(fht_map2.vertex):
                    if isinstance(now_vertex_2, Support_Vertex):
                        continue

                    now_similarity = np.dot(now_descriptor, now_vertex_2.descriptor)
                    #对于相似性超过阈值的直接进行相对位姿变换估计
                    if now_similarity > self.similarity_th:
                        estimated_RP = self.single_RP_estimation(robot1_index,robot2_index,virtual_vertex,now_vertex_2)
                        if not (estimated_RP is None):
                            obtain_new_est = True
                   
        #对子图内部做优化
        if obtain_new_est:
            sub_graphs = self.tf_graph_manager.obtain_sub_connected_graph()
            for now_subgraph in sub_graphs:
                if np.sum(now_subgraph) == 1:
                    continue
                #now subgraph is a list
                result = self.topo_optimize(now_subgraph)

    def single_RP_estimation(self,robot1_index,robot2_index,vertex1,vertex2):
        # print("single estimation")
        final_R, final_t = self.single_estimation(vertex2.local_laserscan_angle,vertex1.local_laserscan_angle) #估计1->2的坐标变换
        if final_R is None or final_t is None:
            return None
        else:
            estimated_pose = [final_t[0][0],final_t[1][0], math.atan2(final_R[1,0],final_R[0,0])]
            #obtain an estimation from robot1 to robot2, and the pose

            self.estimated_vertex_pose[(robot1_index,robot2_index)].append([vertex1.pose,vertex2.pose,estimated_pose]) #ICP输出的结果是robot1 -> robot2
            # print(f"robot {robot1_index+1} -> robot {robot2_index+1}",self.estimated_vertex_pose[(robot1_index,robot2_index)])

            T_map2_odom2 = change_frame([0,0,0],vertex2.pose) #odom2坐标系下map2的坐标
            T_odom2_map1 = change_frame(estimated_pose,change_frame([0,0,0],vertex1.pose))
            T_map2_map1 = change_frame(T_map2_odom2,change_frame([0,0,0],T_odom2_map1))

            #create tf graph
            self.tf_graph_manager.add_tf_trans(robot1_index,robot2_index,T_map2_map1)

            return T_map2_map1
    
    def topo_optimize(self,robot_index_list):
        #self.estimated_vertex_pose.append([self.self_robot_name, vertex.robot_name,svertex.id,vertex.id,pose])
        # This part should update self.map_frame_pose[vertex.robot_name];self.map_frame_pose[vertex.robot_name][0] R33;[1]t 31
        # print(robot_index_list)
        #首先添加这个group内所有机器人的位姿
        first_flag = True
        first_robot = -1
        trans_data = ""
        for index in range(len(robot_index_list)):
            robot_in_group_flag = robot_index_list[index]
            if not robot_in_group_flag:#如果当前机器人不在集群内，直接跳过
                continue
            if first_flag:
                trans_data+="VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n".format(index,0,0,0)
                first_robot = index
                first_flag = False
            else:
                init_guess_of_trans = self.tf_graph_manager.get_relative_trans(first_robot,index)
                trans_data+="VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n".format(index,init_guess_of_trans[0],init_guess_of_trans[1],init_guess_of_trans[2])

        #添加边
        now_trust = 1
        for robot1_index in range(self.robot_num):
            if not robot_index_list[robot1_index]:
                continue

            for robot2_index in range(self.robot_num):
                if not robot_index_list[robot2_index]:
                    continue

                now_estimation_list = self.estimated_vertex_pose[(robot1_index,robot2_index)] #需要进制转化一下，换成角度制
                for now_est in now_estimation_list:
                    trans_data+="EDGE_SE2 {} {} ".format(robot1_index,robot2_index)
                    for j in range(3):
                        for k in range(2):
                            trans_data += " {:.6f} ".format(now_est[j][k]) #add edge information
                        trans_data += " {:.6f} ".format(now_est[j][2]/math.pi * 180) #对于角度转化到角度制
                    trans_data += " {:.6f} 0 0 {:.6f} 0 {:.6f}\n".format(now_trust,now_trust,now_trust)
        
        # print(trans_data)
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
        poses_optimized[:,-1] = poses_optimized[:,-1]
        # print("Optimized Result: ",poses_optimized)
        result_index = 1
        for index in range(len(robot_index_list)):
            if not robot_index_list[index]:
                continue
            if index == first_robot:
                continue
            now_meeted_robot_pose = poses_optimized[result_index,:]
            print(f"Optimized Pose from /robot{first_robot+1}/map to /robot{index+1}/map: ", now_meeted_robot_pose)
            

            self.tf_graph_manager.add_tf_trans(first_robot,index,now_meeted_robot_pose.tolist())
            result_index += 1
    
    def angle_laser_to_xy(self, laser_angle):
        # input: laser_angle : 1*n array
        # output: laser_xy : 2*m array with no nan
        angle_min = 0
        angle_increment = 0.017501922324299812
        laser_cos = np.cos(angle_min + angle_increment * np.arange(len(laser_angle)))
        laser_sin = np.sin(angle_min + angle_increment * np.arange(len(laser_angle)))
        laser_scan_cos_sin = np.stack([laser_cos, laser_sin])
        valid_indices = np.isfinite(laser_angle)
        laser_xy  = np.array(laser_angle[valid_indices] * laser_scan_cos_sin[:,valid_indices])
        return laser_xy  

    def single_estimation(self,scan1,scan2):
        #vertex1: received map 
        #vertex2: vertex of robot 1
        #return a 3x3 rotation matrix and a 3x1 tranform vector
        vertex_laser = scan1
        now_laser = scan2
        #do ICP to recover the relative pose
        now_laser_xy = self.angle_laser_to_xy(now_laser)
        vertex_laser_xy = self.angle_laser_to_xy(vertex_laser)

        pc1 = np.vstack((now_laser_xy, np.zeros(now_laser_xy.shape[1])))
        pc2 = np.vstack((vertex_laser_xy, np.zeros(vertex_laser_xy.shape[1])))

        processed_source = o3d.geometry.PointCloud()
        pc2_offset = copy.deepcopy(pc2)
        pc2_offset[2,:] -= 0.05
        processed_source.points = o3d.utility.Vector3dVector(np.vstack([pc2.T,pc2_offset.T]))

        processed_target = o3d.geometry.PointCloud()
        pc1_offset = copy.deepcopy(pc1)
        pc1_offset[2,:] -= 0.05
        processed_target.points = o3d.utility.Vector3dVector(np.vstack([pc1.T,pc1_offset.T]))

        final_R, final_t = global_icp(processed_source, processed_target, None, vis=0)

        return final_R, final_t

        




                        


    
if __name__ == '__main__':
    rospy.init_node('multi_robot_explore')
    robot_num = rospy.get_param("~robot_num")

    real_robot_explore_manager = multi_rendezvous_manager(robot_num)

    # robot_start_move_pub = rospy.Publisher('/need_new_goal', Int32, queue_size=100)
    # for i in range(robot_num):
    #     robot_index_msg = Int32
    #     robot_index_msg.data = i
    #     robot_start_move_pub.publish(robot_index_msg)
    #     print("publish a msg")

    rospy.spin()

    