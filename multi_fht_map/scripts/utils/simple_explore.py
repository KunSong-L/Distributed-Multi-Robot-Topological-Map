#!/usr/bin/python3.8
import rospy
import tf
from std_msgs.msg import  Int32
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import  PoseStamped, Point
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from robot_function import detect_frontier,sparse_point_cloud
from sklearn.cluster import DBSCAN
from collections import Counter


class RobotNode:
    def __init__(self, robot_name,use_clustered_frontier = False):#输入当前机器人name
        self.self_robot_name = robot_name
        self.robot_index = int(robot_name[-1])-1 #start from 0
        self.perform_rend_flag = False
        #robot data
        self.pose = [0,0,0] # x y yaw angle in degree
        self.erro_count = 0
        self.goal = np.array([])

        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.map_origin = [0,0]
        self.use_clustered_frontier = use_clustered_frontier
        
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
        self.last_goal_time = rospy.Time.now().to_sec()
        self.max_goal_duration = 10
        self.start_follow_path_flag = False
        self.path_point = None
        rospy.Subscriber(robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)

        rospy.Subscriber(robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)
        rospy.Subscriber(robot_name+"/move_base/status", GoalStatusArray, self.follow_path, queue_size=1)
        


        self.actoinclient.wait_for_server()

    def change_goal(self,move_goal,extend_length = 4):
        goal_message, goal_marker, self.goal = self.get_move_goal_and_marker(self.self_robot_name,move_goal,extend_length)#offset = 0
        self.actoinclient.send_goal(goal_message)
        self.goal_pub.publish(goal_marker)
        self.last_goal_time = rospy.Time.now().to_sec()
    
    def get_move_goal_and_marker(self, robot_name, goal,extend_length):
        #next angle should be next goal direction
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()

        goal_vector = np.array(goal - self.pose[0:2])

        length_vector = np.linalg.norm(goal_vector)
        if length_vector == 0:
            extended_goal_vector = goal_vector* extend_length
        else:
            extended_goal_vector = goal_vector / np.linalg.norm(goal_vector) * extend_length

        new_goal = goal + extended_goal_vector #把目标向前延伸一点
        new_goal_pixel = np.array([int((new_goal[0] - self.map_origin[0])/self.map_resolution), int((new_goal[1] - self.map_origin[1])/self.map_resolution)])
        local_range = 5
        local_map_near_goal = self.global_map[new_goal_pixel[1]-local_range:new_goal_pixel[1]+local_range, new_goal_pixel[0]-local_range:new_goal_pixel[0]+local_range]

        if self.global_map[new_goal_pixel[1],new_goal_pixel[0]] != 255 or np.any(local_map_near_goal == 100):
            new_goal = goal #如果new goal是一个已经探索过得地方，那么就之间发原来的位置

        move_direction = np.arctan2(goal_vector[1],goal_vector[0])
        orientation = R.from_euler('z', move_direction, degrees=False).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]
        pose = Point()
        pose.x = new_goal[0]
        pose.y = new_goal[1]
        goal_message.target_pose.pose.position = pose

        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.pose.orientation.x = orientation[0]
        goal_marker.pose.orientation.y = orientation[1]
        goal_marker.pose.orientation.z = orientation[2]
        goal_marker.pose.orientation.w = orientation[3]
        pose = Point()
        pose.x = new_goal[0]
        pose.y = new_goal[1]
        goal_marker.pose.position = pose
        return goal_message, goal_marker, new_goal

    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        try:
            self.tf_listener2.waitForTransform(self.self_robot_name+"/map", self.self_robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
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
        if self.use_clustered_frontier:
            for frontier in self.clustered_frontier:
                point_msg = Point()
                point_msg.x = frontier[0]
                point_msg.y = frontier[1]
                point_msg.z = 0.3
                frontier_marker.points.append(point_msg)
        else:
            for frontier in self.total_frontier:
                point_msg = Point()
                point_msg.x = frontier[0]
                point_msg.y = frontier[1]
                point_msg.z = 0.2
                frontier_marker.points.append(point_msg)
        
        self.frontier_publisher.publish(frontier_marker)

    def map_grid_callback(self, data):
        #如果当前没有目标，就发送需要VRP assign的消息
        if len(self.goal) == 0:
            self.need_a_new_goal()
        #generate grid map and global grid map
        shape = (data.info.height, data.info.width)
        self.map_origin  = [data.info.origin.position.x,data.info.origin.position.y]
        
        self.global_map_tmp = np.asarray(data.data).reshape(shape)
        self.global_map_tmp[np.where(self.global_map_tmp==-1)] = 255
        self.global_map = copy.deepcopy(self.global_map_tmp)

        try:
            #detect frontier
            #only detect frontier in local map
            current_loc_pixel = [0,0]
            local_range = int(10/self.map_resolution)
            current_loc_pixel[0] = int((self.pose[1] - data.info.origin.position.y)/data.info.resolution)
            current_loc_pixel[1] = int((self.pose[0] - data.info.origin.position.x)/data.info.resolution)
            local_grid_map = copy.deepcopy(self.global_map[current_loc_pixel[0]-local_range:current_loc_pixel[0]+local_range, current_loc_pixel[1]-local_range:current_loc_pixel[1]+local_range])
            detected_frontiers = detect_frontier(local_grid_map)
            detected_frontiers_origin = detected_frontiers + np.array([current_loc_pixel[1] - local_range,current_loc_pixel[0] - local_range])

            current_frontier =  detected_frontiers_origin* self.map_resolution + np.array(self.map_origin)

            self.total_frontier = np.vstack((self.total_frontier, current_frontier))
            if self.use_clustered_frontier:
                ds_size = 0.2
            else:
                ds_size = 0.5
            self.total_frontier = sparse_point_cloud(self.total_frontier, ds_size)
        except:
            print("error in detect frontier")
        
        self.update_robot_pose()
        self.update_frontier()

        #如果上一个目标已经过去太久了，则更新一个
        if not self.perform_rend_flag:
            now_time = rospy.Time.now().to_sec()
            if (now_time - self.last_goal_time) > self.max_goal_duration:
                self.need_a_new_goal()

        if self.use_clustered_frontier:
            self.clustered_frontier = self.frontier_cluster(self.total_frontier,0.5,3).reshape((-1,2))
        else:
            self.clustered_frontier = self.total_frontier

        self.visulize_frontier()

    def move_base_status_callback(self, data):
        if len(data.status_list) == 0:
            return

        status = data.status_list[-1].status
        if status == 3:
            #goal reached
            self.need_a_new_goal()
        elif status > 3:
            self.erro_count +=1

        if self.erro_count >= 1:
            if len(self.goal) != 0:
                print("robot ",self.robot_index +1, " add a dead area",self.goal)
                self.dead_area.append(copy.deepcopy(self.goal))
            self.erro_count = 0
            self.need_a_new_goal()


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
        if self.use_clustered_frontier:
            frontier_position = np.array([int((pose_in_world[0] - self.map_origin[0])/self.map_resolution), int((pose_in_world[1] - self.map_origin[1])/self.map_resolution)])
            expored_range = 4
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            for now_area_pos in self.dead_area:
                if np.linalg.norm(now_area_pos - pose_in_world) < 2: #对于以前不可达的frontier直接删掉
                    return True
            expored_range = 6
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range, frontier_position[0]-expored_range:frontier_position[0]+expored_range]
            if np.any(temp_map == 100):# delete near obstcal frontier
                return True
                
            if np.all(temp_map == 0): #Only explored area in this region
                return True
            else:
                return False
        else:
            expored_range = 1
            frontier_position = np.array([int((pose_in_world[0] - self.map_origin[0])/self.map_resolution), int((pose_in_world[1] - self.map_origin[1])/self.map_resolution)])
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.logical_not(np.any(temp_map == 255)): #unkown place is not in this point
                return True

            expored_range = 6
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range, frontier_position[0]-expored_range:frontier_position[0]+expored_range]
            if np.any(temp_map == 100):# delete near obstcal frontier
                return True

            for now_area_pos in self.dead_area:
                if np.linalg.norm(now_area_pos - pose_in_world) < 2: #对于以前不可达的frontier直接删掉
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
        try:
            self.total_frontier = np.delete(self.total_frontier, delete_index, axis = 0)
        except:
            pass


        #goal in map frame
        now_goal = self.goal
        
        if now_goal.size > 0:
            # near the goal 
            if np.linalg.norm(np.array(now_goal[0:2]) - np.array(self.pose[0:2])) < 1:
                self.need_a_new_goal()
                return
            frontier_position = np.array([int((now_goal[0] - self.map_origin[0])/self.map_resolution), int((now_goal[1] - self.map_origin[1])/self.map_resolution)])
            expored_range = 4
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.any(temp_map == 100):
                # print("Target near obstacle! Change another goal!")
                self.need_a_new_goal()
                return
            
            expored_range = 1
            temp_map = self.global_map[frontier_position[1]-expored_range:frontier_position[1]+expored_range+1, frontier_position[0]-expored_range:frontier_position[0]+expored_range+1]
            if np.logical_not(np.any(temp_map == 255)):
                # print("Target is an explored point! Change another goal!")
                self.need_a_new_goal()
                return

    def follow_path(self,data):
        if not self.start_follow_path_flag:
            return

        path_point = copy.deepcopy(self.path_point)
        goal_reached = False
        now_goal_index = 0
        self.change_goal(path_point[now_goal_index], 0)
        while(not goal_reached):
            now_robot_pose = np.array(self.pose)[0:2]

            #检测是否有以后的可去的点
            find_new_target = False
            for i in range(len(path_point)-1,now_goal_index,-1):#逆序索引

                target_vertex_pose_pixel = (np.array(path_point[i])- np.array(self.map_origin))/self.map_resolution
                x = int(target_vertex_pose_pixel[0])
                y = int(target_vertex_pose_pixel[1])
                height, width = self.global_map.shape
                if x > 0 and x < width and y > 0 and y < height and self.global_map[y,x] == 0: #如果目标可见
                    #目标点不要过远
                    if np.linalg.norm(np.array(path_point[i]) - now_robot_pose) < 7:
                        now_target_vertex = i 
                        find_new_target = True
                        break

            if find_new_target:
                now_goal_index=now_target_vertex
                self.change_goal(path_point[now_goal_index], 0)
                continue
                
            now_path_point = np.array(path_point[now_goal_index])[0:2]
            
            if now_goal_index == len(path_point)-1:
                if np.linalg.norm(now_robot_pose - now_path_point) < 2:
                    print(self.self_robot_name + "goal reached")
                    break
            else:
                if np.linalg.norm(now_robot_pose - now_path_point) < 0.5:
                    now_goal_index+=1
                    self.change_goal(path_point[now_goal_index], 0)
            
            rospy.sleep(0.1)
        self.start_follow_path_flag= False

            


