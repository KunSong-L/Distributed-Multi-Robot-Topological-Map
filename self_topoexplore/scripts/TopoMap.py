import numpy as np
import cv2
import math
from networkx.generators import social
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import copy
import rospy

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

    def __init__(self, robot_name=None, id=None, pose=None, descriptor=None, localMap=None, local_image=None, local_laserscan=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.descriptor = descriptor
        self.localMap = localMap
        self.local_laserscan = local_laserscan #2*n array
        self.navigableDirection = []
        self.frontierPoints = []
        self.frontierDistance = []
        self.local_image = local_image


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
    
    def insert(self, vertex=None, edge=None) -> None:
        self.vertex.append(vertex)
        self.edge.append(edge)

    def add(self, vertex=None, last_vertex=-1, current_node=None) -> None:
        #current_node: now match node
        #last_vertex: last added vertex
        matched_flag = 0
        if current_node != None:# the initial value of current node is None, so this line means that current node is initialized
            temp_name = current_node.robot_name # 可以认为是上一个创建的点
            temp_id = current_node.id
        max_score = 0
        #与自己的vertex进行匹配，检验是否为同一点
        for items in self.vertex:
            score = np.dot(vertex.descriptor.T, items.descriptor) #转置之后点积，这里应该不用trans也可以
            point1 = np.array([vertex.pose[0], vertex.pose[1]])
            point2 = np.array([items.pose[0], items.pose[1]])
            dis = np.linalg.norm(point1 - point2)
            if score > self.threshold or dis < 1.5:  #找到距离最近的点，以及一个匹配的点
                matched_flag = 1
                if score > max_score: #find best match
                    max_score = score
                    current_node = items
        
        if matched_flag == 0:
            self.vertex_id += 1
            vertex.id = self.vertex_id
            current_node = vertex #add a new vertex
            self.vertex.append(vertex)
            self.x = np.concatenate((self.x, [vertex.pose[0]]), axis=0) # the x and y of vertex
            self.y = np.concatenate((self.y, [vertex.pose[1]]), axis=0)
            self.center = np.array([np.mean(self.x), np.mean(self.y)]) # center of now whole vertex
            if last_vertex >= 0:
                link = [[temp_name, temp_id], [vertex.robot_name, vertex.id]]
                self.edge.append(Edge(id=self.edge_id, link=link))
            self.edge_id += 1
        else:# matched
            if current_node.robot_name != temp_name or current_node.id != temp_id:#类似于形成自我回环的情况，这里是可以做后端优化的
                #matched with other robots' vertex
                link = [[temp_name, temp_id], [current_node.robot_name, current_node.id]] # connect
                self.edge.append(Edge(id=self.edge_id, link=link))
                self.edge_id += 1
        
        return self.vertex_id, current_node, matched_flag
    
    
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
            