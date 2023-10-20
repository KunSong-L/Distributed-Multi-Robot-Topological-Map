#!/usr/bin/python3.8
import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import  PoseStamped, Point
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import copy
from robot_function import *


debug_path = "/home/master/debug/test1/"
save_result = False

class robot_expore:
    def __init__(self, robot_name):#输入当前机器人name
        self.self_robot_name = robot_name
        
        #robot data
        self.pose = [0,0,0] # x y yaw angle in degree
        self.init_map_angle_ready = 0
        self.map_orientation = None
        self.map_angle = None #Yaw angle of map
        self.current_loc_pixel = [0,0]
        self.erro_count = 0
        self.goal = np.array([])
        self.grid_map_ready = 0
        self.tf_transform_ready = 0
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        self.allow_robot_move = False # robot will start explore until receive an alow move flag
        
        self.vis_color = np.array([[0xFF, 0x7F, 0x51], [0xD6, 0x28, 0x28],[0xFC, 0xBF, 0x49],[0x00, 0x30, 0x49],[0x1E, 0x90, 0xFF]])/255.0
        # get tf
        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None
        #move base
        self.actoinclient = actionlib.SimpleActionClient(self.self_robot_name+'/move_base', MoveBaseAction)
        
        self.total_frontier = np.array([],dtype=float).reshape(-1,2)
        self.finish_explore = False

        self.goal_pub = rospy.Publisher(self.self_robot_name+"/goal", PoseStamped, queue_size=1)

        self.frontier_publisher = rospy.Publisher(self.self_robot_name+'/frontier_points', Marker, queue_size=1)
        rospy.Subscriber(self.self_robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)

        rospy.Subscriber(self.self_robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)


        self.actoinclient.wait_for_server()

    def change_goal(self):
        # move goal:now_pos + basic_length+offset;  now_angle + nextmove
        if len(self.total_frontier) == 0:
            return
        if not self.allow_robot_move:
            return
        move_goal = self.choose_exp_goal()
        goal_message, self.goal = self.get_move_goal(self.self_robot_name,move_goal )#offset = 0
        goal_marker = self.get_goal_marker(self.self_robot_name, move_goal)
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)

    def choose_exp_goal(self):
        #在frontier里面计算代价函数
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

    def get_move_goal(self, robot_name, goal)-> MoveBaseGoal():
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()
        move_dir = goal - np.array(self.pose[0:2])
        move_angle = np.arctan2(move_dir[1],move_dir[0])
        orientation = R.from_euler('z', move_angle, degrees=True).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]
        # dont decide which orientation to choose 
        # goal_message.target_pose.pose.orientation.x = 0
        # goal_message.target_pose.pose.orientation.y = 0
        # goal_message.target_pose.pose.orientation.z = 0
        # goal_message.target_pose.pose.orientation.w = 1
        pose = Point()
        pose.x = goal[0]
        pose.y = goal[1]
        goal_message.target_pose.pose.position = pose

        return goal_message, goal

    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        self.tf_listener2.waitForTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
        try:
            self.tf_transform, self.rotation = self.tf_listener2.lookupTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow)
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

    def visulize_vertex(self):
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
        for frontier in self.total_frontier:
            point_msg = Point()
            point_msg.x = frontier[0]
            point_msg.y = frontier[1]
            point_msg.z = 0.2
            frontier_marker.points.append(point_msg)
        self.frontier_publisher.publish(frontier_marker)
        # --------------finish visualize frontier---------------

    def map_grid_callback(self, data):
        
        #generate grid map and global grid map
        range = int(6/self.map_resolution)
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)
        timenow = rospy.Time.now()
        #robot1/map->robot1/base_footprint
        self.tf_listener.waitForTransform(data.header.frame_id, self.self_robot_name+"/base_footprint", timenow, rospy.Duration(0.5))

        tf_transform, rotation = self.tf_listener.lookupTransform(data.header.frame_id, self.self_robot_name+"/base_footprint", timenow)
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
        self.change_goal()
        self.visulize_vertex()
        #保存图片
        if save_result:
            temp = self.global_map[max(self.current_loc_pixel[0]-range,0):min(self.current_loc_pixel[0]+range,shape[0]), max(self.current_loc_pixel[1]-range,0):min(self.current_loc_pixel[1]+range, shape[1])]
            temp[np.where(temp==-1)] = 125
            cv2.imwrite(debug_path+self.self_robot_name + "_local_map.jpg", temp)
            cv2.imwrite(debug_path+self.self_robot_name +"_global_map.jpg", self.global_map)

    def move_base_status_callback(self, data):
        self.update_robot_pose()
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

if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('robot_explore')
    robot_name = rospy.get_param("~robot_name")

    node = robot_expore(robot_name)

    rospy.spin()