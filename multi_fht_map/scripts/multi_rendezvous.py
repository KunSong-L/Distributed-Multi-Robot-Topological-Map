#!/usr/bin/python3.8
import rospy
import tf
from std_msgs.msg import String, Int32
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import  PoseStamped, Point
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import copy
from robot_function import *
from utils.solveVRP import VRP_solver
from utils.tf_graph_manager import multi_robot_tf_manager
from sklearn.cluster import DBSCAN
from collections import Counter

save_result = False

class RobotNode:
    def __init__(self, robot_name):#输入当前机器人name
        self.self_robot_name = robot_name
        self.robot_index = int(robot_name[-1])-1 #start from 0
        
        #robot data
        self.pose = [0,0,0] # x y yaw angle in degree
        self.erro_count = 0
        self.goal = np.array([])

        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        
        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        
        #move base
        self.dead_area = [] #record goal that was not reachable, see it as a dead area
        self.actoinclient = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)
        self.total_frontier = np.array([],dtype=float).reshape(-1,2)
        self.clustered_frontier = np.array([],dtype=float).reshape(-1,2)
        self.finish_explore = False

        self.goal_pub = rospy.Publisher(robot_name+"/goal", PoseStamped, queue_size=1)

        self.frontier_publisher = rospy.Publisher(robot_name+'/frontier_points', Marker, queue_size=1)
        self.need_new_goal_pub = rospy.Publisher('/need_new_goal', Int32, queue_size=100) #是否需要一个新的前沿点

        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)

        rospy.Subscriber(
            robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)


        self.actoinclient.wait_for_server()

    def change_goal(self,move_goal):
        goal_message, goal_marker, self.goal = self.get_move_goal_and_marker(self.self_robot_name,move_goal )#offset = 0
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)
    
    def get_move_goal_and_marker(self, robot_name, goal):
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()

        goal_vector = np.array(np.array(goal - self.pose[0:2]))
        move_direction = np.arctan2(goal_vector[1],goal_vector[0])
        orientation = R.from_euler('z', move_direction, degrees=False).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]
        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_message.target_pose.pose.position = pose

        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.pose.orientation.x = orientation[0]
        goal_marker.pose.orientation.y = orientation[1]
        goal_marker.pose.orientation.z = orientation[2]
        goal_marker.pose.orientation.w = orientation[3]
        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_marker.pose.position = pose
        return goal_message, goal_marker, goal

    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        self.tf_listener2.waitForTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
        try:
            self.tf_transform, self.rotation = self.tf_listener2.lookupTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow)
            self.pose[0] = self.tf_transform[0]
            self.pose[1] = self.tf_transform[1]
            self.pose[2] = R.from_quat(self.rotation).as_euler('xyz', degrees=True)[2]

        except:
            pass

    def visulize_frontier(self):
        # ----------visualize frontier------------
        frontier_marker = Marker()
        now = rospy.Time.now()
        frontier_marker.header.frame_id = self.self_robot_name + "/map"
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
        # for frontier in self.total_frontier:
        #     point_msg = Point()
        #     point_msg.x = frontier[0]
        #     point_msg.y = frontier[1]
        #     point_msg.z = 0.2
        #     frontier_marker.points.append(point_msg)
        for frontier in self.clustered_frontier:
            point_msg = Point()
            point_msg.x = frontier[0]
            point_msg.y = frontier[1]
            point_msg.z = 0.3
            frontier_marker.points.append(point_msg)
        self.frontier_publisher.publish(frontier_marker)
        # --------------finish visualize frontier---------------

    def map_grid_callback(self, data):
        
        #generate grid map and global grid map
        shape = (data.info.height, data.info.width)
        self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
        
        self.global_map_tmp = np.asarray(data.data).reshape(shape)
        self.global_map_tmp[np.where(self.global_map_tmp==-1)] = 255
        self.global_map = copy.deepcopy(self.global_map_tmp)

        try:
            #detect frontier
            current_frontier = detect_frontier(self.global_map) * self.map_resolution + np.array(self.map_origin)
            self.total_frontier = np.vstack((self.total_frontier, current_frontier))
            ds_size = 0.2
            self.total_frontier = sparse_point_cloud(self.total_frontier, ds_size)
        except:
            pass
        self.update_robot_pose()
        self.update_frontier()
        self.clustered_frontier = self.frontier_cluster(self.total_frontier)
        self.visulize_frontier()

    def move_base_status_callback(self, data):
        try:
            status = data.status_list[-1].status
        # print(status)
        
            if status >= 3:
                self.erro_count +=1
            if self.erro_count >= 3:
                if len(self.goal) != 0:
                    self.dead_area.append(copy.deepcopy(self.goal))
                self.erro_count = 0
                self.need_a_new_goal()
        except:
            pass

    def frontier_cluster(self,frontiers, cluser_eps=0.7, cluster_min_samples=7):
        # input: frontier; DBSCAN eps; DBSCAN min samples
        # output: clustered frontier
        dbscan = DBSCAN(eps=cluser_eps, min_samples=cluster_min_samples)
        labels = dbscan.fit_predict(frontiers)
        label_counts = Counter(labels)
        clustered_frontier = []
        for label_key in label_counts.keys():
            if label_key == -1:
                continue
            label_index = labels==label_key
            frontier_center = np.sum(frontiers[label_index,:],axis=0)/label_counts[label_key]
            clustered_frontier.append(frontier_center)

        return np.array(clustered_frontier)

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

    def need_a_new_goal(self):
        new_goal_msg = Int32()
        new_goal_msg.data = int(self.robot_index)
        self.need_new_goal_pub.publish(new_goal_msg) #find a better path

    def update_frontier(self):
        #负责删除一部分前沿点
        position = self.pose
        position = np.array([position[0], position[1]])
        #delete unexplored direction based on distance between now robot pose and frontier point position
        delete_index = []
        for index, frontier in enumerate(self.total_frontier):
            if self.is_explored_frontier(frontier):
                delete_index.append(index)
            for now_area_pos in self.dead_area:
                if np.linalg.norm(now_area_pos - frontier) < 2: #对于以前不可达的frontier直接删掉
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
                self.need_a_new_goal()
                return
            
            expored_range = 1
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.logical_not(np.any(temp_map == 255)):
                # print("Target is an explored point! Change another goal!")
                self.need_a_new_goal()
                return



class robot_explore_manager():
    def __init__(self, robot_num):#the maxium robot num
        self.robot_num = robot_num
        robot_list_name = []
        for rr in range(robot_num):
            robot_list_name.append("robot"+str(rr+1))
        
        self.RobotNode_list = []
        for now_robot in robot_list_name:
            self.RobotNode_list.append(RobotNode(now_robot))
        
        self.vrp_solver = VRP_solver(None,None)

        self.tf_graph_manager = multi_robot_tf_manager(robot_num)

        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        self.global_frontier_publisher = rospy.Publisher('/global_frontier_points', Marker, queue_size=1)
        rospy.Subscriber('/need_new_goal', Int32, self.vrp_assign, queue_size=100)

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
        robot_pose = np.array([]).reshape((-1,2))
        for index in range(self.robot_num):
            if now_sub_graph[index] != 1:
                continue
            if target_frame == -1:
                target_frame = index
            now_robot_frontier = self.RobotNode_list[index].clustered_frontier
            #统一坐标系到target_frame
            frame_trans = self.tf_graph_manager.get_relative_trans(index,target_frame)
            target_frame_frontier = change_frame_multi(now_robot_frontier,frame_trans)
            total_frontier = np.vstack((total_frontier, target_frame_frontier))

            #把机器人的位置也改变一下坐标系
            robot_pose = self.RobotNode_list[index].pose
            target_frame_pose = change_frame(robot_pose,frame_trans)
            robot_pose = np.vstack((robot_pose, target_frame_pose))

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

        #分配前沿点
        self.vrp_solver.robot_pose = robot_pose[:,0:2] #只采用距离代价，先不考虑旋转
        real_total_frontier = total_frontier[not_expored_index,:]
        self.vrp_solver.points = real_total_frontier #仅分配还没探索过的
        result_path,path_length = self.vrp_solver.solveVRP()

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
    
    def visulize_frontier(self,robot_name,frontiers):
        # ----------visualize frontier------------
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

    real_robot_explore_manager = robot_explore_manager(robot_num)

    rospy.spin()