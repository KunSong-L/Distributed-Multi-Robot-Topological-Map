import numpy as np
import cv2
import math
from networkx.generators import social
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import copy
import rospy
from robot_function import calculate_entropy

def if_frontier(window):
    if 100 in window: # 障碍物
        return False
    if 0 not in window: # 可通过
        return False
    if 255 not in window: # 未知
        return False

    return True

def get_frontier_points(map, resolution=0.01) -> list:
    shape = map.shape
    kernel_size = int(0.2/resolution)
    step_size = kernel_size//2
    min_num = 10
    frontier_points = []
    for i in range(0,shape[0]-kernel_size,step_size):
        for j in range(0,shape[1]-kernel_size,step_size):
            if if_frontier(map[i:i+kernel_size, j:j+kernel_size]): #找到已知和未知的边界
                frontier_points.append([i+step_size, j+step_size])
    if frontier_points:# not empty
        dbscan = DBSCAN(eps=kernel_size, min_samples=4).fit(frontier_points)#聚类
        lables = np.unique(dbscan.labels_)# 获取有几类
        points_list = [list() for i in range(len(lables))]#获取每一类具体有多少点
    centers = []
    
    for i in range(len(frontier_points)):
        points_list[dbscan.labels_[i]].append(frontier_points[i])#把每个点加进对应类里面去
    
    if frontier_points:
        for point in points_list:
            x,y = zip(*point)
            if len(x) < min_num:#直接过滤掉小于min_num个边界点的情况
                continue
            center_tmp = (int(np.mean(x)), int(np.mean(y)))
            centers.append(center_tmp)
    
    return centers

class Vertex:
    def __init__(self, robot_name=None, id=None, pose=None, descriptor=None, localMap=None, local_image=None, local_laserscan_angle=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.descriptor = descriptor
        self.localMap = localMap
        self.local_laserscan_angle = local_laserscan_angle
        # self.local_laserscan = local_laserscan #2*n array
        self.navigableDirection = []
        self.frontierPoints = []
        self.frontierDistance = []
        self.local_image = local_image
        self.descriptor_infor = 0
        self.local_free_space_rect = [0,0,0,0] #x1,y1,x2,y2 in map frame  ; x1< x2,y1 <y2
        
        if descriptor is not None:
            self.descriptor_infor = calculate_entropy(descriptor)

class Support_Vertex:
    def __init__(self, robot_name=None, id=None, pose=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.local_free_space_rect = [0,0,0,0]

class Edge:
    
    def __init__(self, id, link) -> None:
        self.id = id
        self.link = link # [[last_robot_name, last_robot_id], [now_robot_name, now_vertex_id]]


class TopologicalMap:
    
    def __init__(self, robot_name='1', threshold=0.8) -> None:
        self.robot_name = robot_name
        self.vertex = list()#保存了所有节点
        self.edge = list()
        self.threshold = threshold
        self.vertex_id = -1
        self.edge_id = 0
        self.unexplored_points = list()
        self.x = np.array([])
        self.y = np.array([])
        self.center = None
        self.center_dict = dict()
        self.offset_angle = 0
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
        self.rotation = np.eye(3) #rotation from this topomap to robot1 frame
        self.trans_vector = np.array([0,0,0])
    
    def insert(self, vertex=None, edge=None) -> None:
        self.vertex.append(vertex)
        self.edge.append(edge)

    def add(self, vertex=None) -> None:
        self.vertex_id += 1
        vertex.id = self.vertex_id
        current_node = vertex #add a new vertex
        self.vertex.append(vertex)

        return self.vertex_id, current_node
    
    def change_topomap_frame(self, map_frame_pose):
        #topomap: FHT-Map in other frame
        #map_frame_pose: map_frame_pose[0] 3x3 rotation matrix of R^robot1_other
        #map_frame_pose: map_frame_pose[1] 3x1 transform vector of t^robot1_other
        now_robot_map_frame_rot = map_frame_pose[0]
        now_robot_map_frame_trans = map_frame_pose[1]
        for vertex in self.vertex:
            #transpose vertex pose
            tmp_pose = now_robot_map_frame_rot @ np.array([vertex.pose[0],vertex.pose[1],0]) + now_robot_map_frame_trans
            vertex.pose[0] = tmp_pose[0]
            vertex.pose[1] = tmp_pose[1]
            #transpose pose of local free space
            p1_local_s = np.append(np.array(vertex.local_free_space_rect[0:2]),0)
            p1_local_s = now_robot_map_frame_rot @ p1_local_s + now_robot_map_frame_trans
            p2_local_s = np.append(np.array(vertex.local_free_space_rect[2:]),0)
            p2_local_s = now_robot_map_frame_rot @ p2_local_s + now_robot_map_frame_trans
            vertex.local_free_space_rect = [p1_local_s[0],p1_local_s[1],p2_local_s[0],p2_local_s[1]]
        
        #update transpose of FHT-Map
        self.rotation = copy.deepcopy(map_frame_pose[0])
        self.trans_vector = copy.deepcopy(map_frame_pose[1])
            

    
    def upgradeFrontierPoints(self, vertex_id=-1, type="new", resolution=0.05):
        picked_vertex = None
        picked_vertex_id = 0
        if type == "new":
            #准备在新加入的节点附近计算一系列前沿点
            for i in range(len(self.vertex)):
                vertex = self.vertex[i]
                if vertex.robot_name == self.robot_name and vertex.id == vertex_id:
                    picked_vertex = vertex
                    picked_vertex_id = i
                    break
        elif type == "old":
            picked_vertex = self.vertex[vertex_id]
            picked_vertex_id = vertex_id
        shape = picked_vertex.localMap.shape
        center = np.array([int(shape[0]/2), int(shape[1]/2)])
        if vertex_id == -1:
            pass
        frontiers = get_frontier_points(picked_vertex.localMap,resolution=self.map_resolution)#返回一系列边界中心

        temp_frontier_dis = []
        temp_frontier_pos = []
        temp_angle = []
        #计算边界点距离中心的距离、在局部坐标系下的位置、角度
        for front in frontiers:
            front = np.array([front[0], front[1]])# in image frame
            dis = np.sqrt(np.sum(np.square(front-center))) * resolution
            frontier_local_frame = np.array([front[1] - center[1], front[0]-center[0] ])*resolution
            angle = math.degrees(math.atan2(frontier_local_frame[1],frontier_local_frame[0]))

            temp_frontier_dis.append(dis)
            temp_frontier_pos.append(frontier_local_frame)# frontier In local frame
            temp_angle.append(angle)
        
        if len(temp_angle) < len(self.vertex[picked_vertex_id].navigableDirection) and type=="old":
            self.vertex[picked_vertex_id].frontierDistance = temp_frontier_dis
            self.vertex[picked_vertex_id].frontierPoints = temp_frontier_pos
            self.vertex[picked_vertex_id].navigableDirection = temp_angle
        #一般只有new这个情况
        if type == "new":
            self.vertex[picked_vertex_id].frontierDistance = temp_frontier_dis
            self.vertex[picked_vertex_id].frontierPoints = temp_frontier_pos
            self.vertex[picked_vertex_id].navigableDirection = temp_angle

        return picked_vertex_id
    
    def plot(self, size, vcolor=(0, 0, 255), ecolor=(0, 255, 0)) -> np.ndarray:
        mapv = np.zeros([size, size, 3], np.uint8)
        pose = dict()
        for vertex in self.vertex:
            pose[vertex.id] = (int((vertex.pose[0]+5)/10 * size), int((5-vertex.pose[1])/10 * size))
            mapv = cv2.circle(mapv, pose[vertex.id], 3, vcolor, -1)
        for edge in self.edge:
            mapv = cv2.line(mapv, pose[edge.link[0]], pose[edge.link[1]], ecolor)
        return mapv
    
    def vertex_num(self) -> int:
        return len(self.vertex)
    
    def displayNavigableDirection(self):
        for vertex in self.vertex:
            print("vertex: ", vertex.id)
            print("direction:")
            for direction in vertex.navigableDirection:
                print(direction)
            