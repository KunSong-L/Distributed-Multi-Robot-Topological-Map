#!/usr/bin/python3.8
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
        self.fhtmap_creater_list = [fht_map_creater(now_robot) for now_robot in robot_list_name]
        

        #拓扑地图融合
        self.mergered_vertex_index = [0 for i in range(robot_num)]
        self.similarity_mat_dict = {(i,j): [] for i in range(robot_num) for j in range(i,robot_num)} #(i,j): mat 

        #机器人运动
        self.vrp_solver = VRP_solver(None,None)

        self.tf_graph_manager = multi_robot_tf_manager(robot_num)

        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        self.global_frontier_publisher = rospy.Publisher('/global_frontier_points', Marker, queue_size=1)
        rospy.Subscriber('/need_new_goal', Int32, self.vrp_assign, queue_size=100)
        rospy.Subscriber('/need_new_goal', Int32, self.fht_map_merger, queue_size=1)
       

    def vrp_assign(self,data):
        #对每个集群内部进行VRP分配前沿点
        #首先获取子图
        sub_graphs = self.tf_graph_manager.obtain_sub_connected_graph()
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

        for now_robot in result_path.keys():
            now_robot_index = now_robot - 1
            if now_robot_index != robot_index: #仅分配请求的机器人
                continue
            robot_path = result_path[now_robot]
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
                C_robot_to_frontier[i,j] = self.fhtmap_creater_list[robot_index].topo_path_planning(robot_pose[i],frontier_pose[j])

        for i in range(n_point):
            for j in range(n_point):
                C_frontier_to_frontier[i,j] = self.fhtmap_creater_list[robot_index].topo_path_planning(frontier_pose[i],frontier_pose[j])
        
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
        for robot1_index in range(robot_num - 1):
            for robot2_index in range(robot1_index + 1,robot_num):
                fht_map1 = self.fhtmap_creater_list[robot1_index].map
                fht_map2 = self.fhtmap_creater_list[robot2_index].map

                main_vertex_index_1 = -1#统计是第几个主节点了
                
                for index1, now_vertex_1 in enumerate(fht_map1.vertex):
                    if isinstance(now_vertex_1, Support_Vertex):
                        continue
                    main_vertex_index_1 += 1
                    if main_vertex_index_1 > len(self.similarity_mat_dict[(robot1_index,robot2_index)]) -1 :
                        self.similarity_mat_dict[(robot1_index,robot2_index)].append([])

                    main_vertex_index_2 = -1
                    for index2, now_vertex_2 in enumerate(fht_map2.vertex):
                        if isinstance(now_vertex_2, Support_Vertex):
                            continue
                        main_vertex_index_2 += 1
                        #如果是一个已经计算过的点，则跳过
                        if index1 <= self.mergered_vertex_index[robot1_index] and index2 <= self.mergered_vertex_index[robot2_index]:
                            continue
                        #对于没计算过的
                        now_similarity = np.dot(now_vertex_1.descriptor.T, now_vertex_2.descriptor)
                        #现在需要把结果添加到self.similarity_mat_dict[(robot1_index,robot2_index)] [i] [j]中
                                                
                        self.similarity_mat_dict[(robot1_index,robot2_index)][main_vertex_index_1].append(now_similarity)
                self.mergered_vertex_index[robot1_index] = len(fht_map1.vertex) -1
                self.mergered_vertex_index[robot2_index] = len(fht_map2.vertex) - 1
        




                        


    
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

    