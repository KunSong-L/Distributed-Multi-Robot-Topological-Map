#!/usr/bin/python3.8
import rospy
from tf2_msgs.msg import TFMessage
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.astar import topo_map_path
import tf
import copy

def change_frame(point_1, T_1_2):
    #根据转换关系把在1坐标系下point转换到1坐标系下
    #T_1_2：[x,y,yaw]格式
    #point_1: 向量[x,y,yaw]或者[x,y]
    #返回：在2坐标系下的point位置
    input_length = len(point_1)
    if input_length==2:
        point_1 = [point_1[0],point_1[1],0]

    R_1_2 = R.from_euler('z', T_1_2[2], degrees=False).as_matrix()
    t_1_2 = np.array([T_1_2[0],T_1_2[1],0]).reshape(-1,1)
    T_1_2 = np.block([[R_1_2,t_1_2],[np.zeros((1,4))]])
    T_1_2[-1,-1] = 1
    
    R_1_point = R.from_euler('z', point_1[2], degrees=False).as_matrix()
    t_1_point = np.array([point_1[0],point_1[1],0]).reshape(-1,1)
    T_1_point = np.block([[R_1_point,t_1_point],[np.zeros((1,4))]])
    T_1_point[-1,-1] = 1

    T_2_point =  np.linalg.inv(T_1_2) @ T_1_point
    rot = R.from_matrix(T_2_point[0:3,0:3]).as_euler('xyz',degrees=False)[2]

    result = [T_2_point[0,-1], T_2_point[1,-1], rot]
    return result[0:input_length]
    
class multi_robot_tf_manager():
    def __init__(self, robot_num, sub_tf_flag = True, use_GT= False):
        #假设需要管理一系列类似的tf
        #如: robot1/map, robot2/map, ..., robotn/map
        #不同map之间的坐标变换可能在某一时刻发布，总体的tf构成一张图
        #该类负责返回图的连同子图，以及可以获取两个frame之间的变换关系

        self.robot_num = robot_num
        self.mat_M_ReEst = np.array([],dtype=np.int16).reshape((robot_num,-1)) #关联矩阵
        self.mat_lap_ReEst = np.array([],dtype=np.int16).reshape((robot_num,-1))
        self.mat_A_ReEst = np.zeros((robot_num,robot_num)) #邻接矩阵
        self.mat_Delta_ReEst = np.zeros((robot_num,robot_num)) #度矩阵

        self.use_GT_flag = use_GT
        self.tf_listener = tf.TransformListener()
        self.sub_grapg_vector = None
        self.gen_new_RP = True #是否增加了一个新的RP
        self.update_RP = True #是否更新了一个新的RP

        #记录已经读取到的相对位姿估计
        self.relative_pose_list = []
        for i in range(robot_num):
            tmp = [None for j in range(robot_num)]
            self.relative_pose_list.append(tmp)
        self.constructed_RP_list = []
        if sub_tf_flag:
            rospy.Subscriber("/tf", TFMessage, self.catch_tf_callback, queue_size=1000, buff_size=52428800)
    
    def catch_tf_callback(self,tf_msg):
        #通过tf这个topic获取相对位姿变换
        #利用tf直接获取实时性比较差
        for transform in tf_msg.transforms:
            # 提取变换信息
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            child_frame_id = transform.child_frame_id
            header = transform.header
            old_frame_id = header.frame_id
            # 打印变换信息
            if "map" in old_frame_id and "map" in child_frame_id:
                relative_pose = [0,0,0]
                relative_pose[0] = translation.x
                relative_pose[1] = translation.y
                rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
                relative_pose[2] = R.from_quat(rotation).as_euler('xyz', degrees=False)[2]
                
                old_robot_index = int(''.join(filter(str.isdigit, old_frame_id))) # 1 for robot1
                child_robot_index = int(''.join(filter(str.isdigit, child_frame_id))) 
                
                self.relative_pose_list[old_robot_index-1][child_robot_index-1] = relative_pose
                print(f"add relative pose from robot {old_robot_index} to {child_robot_index}: {self.relative_pose_list[old_robot_index-1][child_robot_index-1]}")
                #from T 1->2 to T 2->1
                verse_relative_pose = change_frame([0,0,0], relative_pose)
                self.relative_pose_list[child_robot_index-1][old_robot_index-1] = verse_relative_pose

    def add_tf_trans(self,robot1_index,robot2_index,rela_pose):
        #rela_pose: [x,y,yaw]
        if self.use_GT_flag:
            tmptimenow = rospy.Time.now()
            self.tf_listener.waitForTransform(f"/robot{robot1_index+1}"+"/map", f"/robot{robot2_index+1}"+"/map", tmptimenow, rospy.Duration(0.5))
            self.tf_transform, self.rotation = self.tf_listener.lookupTransform(f"/robot{robot1_index+1}"+"/map", f"/robot{robot2_index+1}"+"/map", tmptimenow)
            rela_pose = [0,0,0]
            rela_pose[0] = self.tf_transform[0]
            rela_pose[1] = self.tf_transform[1]
            rela_pose[2] = R.from_quat(self.rotation).as_euler('xyz', degrees=False)[2]
        
        if self.relative_pose_list[robot1_index][robot2_index]==None:
            self.gen_new_RP = True
        self.update_RP = True
        self.relative_pose_list[robot1_index][robot2_index] = rela_pose
        print(rela_pose)
        verse_relative_pose = change_frame([0,0,0], rela_pose)
        self.relative_pose_list[robot2_index][robot1_index] = verse_relative_pose
        


    def obtain_sub_connected_graph(self):
        #获取所有图上的连通子图

        #update A
        if self.gen_new_RP:
            for i in range(self.robot_num):
                for j in range(self.robot_num):
                    now_est = self.relative_pose_list[i][j]
                    if now_est == None:
                        self.mat_A_ReEst[i,j] = 0
                    else:
                        self.mat_A_ReEst[i,j] = 1
            self.mat_Delta_ReEst = np.diag(np.sum(self.mat_A_ReEst,axis=0))
            self.mat_lap_ReEst = self.mat_Delta_ReEst - self.mat_A_ReEst
            # self.mat_lap_ReEst = self.mat_M_ReEst @ self.mat_M_ReEst.T #利用M矩阵求解，第二个定义
            a_small_number = 1e-10
            eigen_vale,eigen_vector = np.linalg.eig(self.mat_lap_ReEst)
            zero_eigen_value = abs(eigen_vale)<a_small_number
            sub_grapg_vector = np.abs(eigen_vector[:,zero_eigen_value])>a_small_number
            self.sub_grapg_vector = sub_grapg_vector
            self.gen_new_RP = False
            return self.sub_grapg_vector.T #m*robot_number bool vector
        else:
            return self.sub_grapg_vector.T #m*robot_number bool vector
        

    def A_mat_to_adj_list(self,A_mat):
        adj_list = dict()
        #把邻接矩阵转化为最短路径算法所需要的格式
        for i in range(self.robot_num):
            for j in range(self.robot_num):
                if A_mat[i,j] == 1:
                    if i not in adj_list.keys():
                        adj_list[i]  = []
                    adj_list[i].append((j, 1))
        return adj_list
    

    def get_relative_trans(self,robot1_index,robot2_index):
        #想要获取T^1_2: robot1 -> robot2的坐标变换
        #robot index start from 0!
        #首先判断是否在一个连通分支上
        if (not self.update_RP) and self.constructed_RP_list[robot1_index][robot2_index] != None:
            return self.constructed_RP_list[robot1_index][robot2_index]
        else:
            subgraphs = self.obtain_sub_connected_graph()
            connected_flag = False
            for now_subgraph in subgraphs:
                if now_subgraph[robot1_index] and now_subgraph[robot2_index]:
                    connected_flag = True
                    break
            
            if not connected_flag:
                print("not connected", subgraphs)
                return None
            adj_list = self.A_mat_to_adj_list(self.mat_A_ReEst)
            topo_map = topo_map_path(adj_list,robot1_index, [robot2_index])
            topo_map.get_path()
            possible_path = topo_map.foundPath[0] #一条连接了两个所需要的相对位姿的路径
            result = [0,0,0] #初始值为0 0 0 
            for i in range(len(possible_path)-1):
                now_trans = self.relative_pose_list[possible_path[i]][possible_path[i+1]]
                result = change_frame(result,now_trans)
            
            
            self.constructed_RP_list = copy.deepcopy(self.relative_pose_list)
            self.constructed_RP_list[robot2_index][robot1_index] = result
            self.constructed_RP_list[robot1_index][robot2_index] = change_frame([0,0,0],result)
            self.update_RP = False
            
            return self.constructed_RP_list[robot1_index][robot2_index]
        


            
        
         