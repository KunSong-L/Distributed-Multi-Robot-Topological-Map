import numpy as np
import cv2
import math
from networkx.generators import social
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import copy


def if_frontier(window):
    if 100 in window: # 障碍物
        return False
    if 0 not in window: # 可通过
        return False
    if 255 not in window: # 未知
        return False

    return True

def get_frontier_points(map, resolution=0.05) -> list:
    shape = map.shape
    kernel_size = 4
    frontier_points = []
    for i in range(shape[0]-kernel_size):
        for j in range(shape[1]-kernel_size):
            if if_frontier(map[i:i+kernel_size, j:j+kernel_size]): #找到已知和未知的边界
                frontier_points.append([i+2, j+2])
    if frontier_points:
        dbscan = DBSCAN(eps=5, min_samples=2).fit(frontier_points)#聚类
        lables = np.unique(dbscan.labels_)# 获取有几类
        points_list = [list() for i in range(len(lables))]#获取每一类具体有多少点
    centers = []
    
    for i in range(len(frontier_points)):
        points_list[dbscan.labels_[i]].append(frontier_points[i])#把每个点加进对应类里面去
    if frontier_points:
        for point in points_list:
            x,y = zip(*point)
            if len(x) < 100:#直接过滤掉小于100个边界点的情况
                continue
            center_tmp = (int(np.mean(x)), int(np.mean(y)))
            centers.append(center_tmp)
    
    return centers

class Vertex:

    def __init__(self, robot_name=None, id=None, pose=None, descriptor=None, localMap=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.descriptor = descriptor
        self.localMap = localMap
        self.navigableDirection = []
        self.frontierPoints = []
        self.frontierDistance = []


class Edge:
    
    def __init__(self, id, link) -> None:
        self.id = id
        self.link = link


class TopologicalMap:
    
    def __init__(self, robot_name='1', threshold=0.8) -> None:
        self.robot_name = robot_name
        self.vertex = list()
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
    
    def insert(self, vertex=None, edge=None) -> None:
        self.vertex.append(vertex)
        self.edge.append(edge)

    def add(self, vertex=None, last_vertex=-1, current_node=None) -> None:
        matched_flag = 0
        if current_node != None:# the initial value of current node is None, so this line means that current node is initialized
            temp_name = current_node.robot_name # 可以认为是上一个创建的点
            temp_id = current_node.id
        max_score = 0
        # print('length of vertes = ', len(self.vertex))
        # lenth of vertex is 1
        #与自己的ertex进行匹配，检验是否为同一点
        for items in self.vertex:
            score = np.dot(vertex.descriptor.T, items.descriptor) #转置之后点积，这里应该不用trans也可以
            point1 = np.array([vertex.pose.pose.position.x, vertex.pose.pose.position.y, vertex.pose.pose.position.z])
            point2 = np.array([items.pose.pose.position.x, items.pose.pose.position.y, items.pose.pose.position.z])
            dis = np.linalg.norm(point1 - point2)
            if score > self.threshold or dis < 2.5:  #找到距离最近的点，以及一个匹配的点
                matched_flag = 1
            if score > max_score:
                max_score = score
                if dis < 2.5:
                    current_node = items # current node是距离当前小于2.5的最匹配的点，定义有点复杂
        
        if matched_flag == 0:
            self.vertex_id += 1
            vertex.id = self.vertex_id
            current_node = vertex #add a new vertex
            self.vertex.append(vertex)
            self.x = np.concatenate((self.x, [vertex.pose.pose.position.x]), axis=0) # the x and y of vertex
            self.y = np.concatenate((self.y, [vertex.pose.pose.position.y]), axis=0)
            self.center = np.array([np.mean(self.x), np.mean(self.y)]) # now robot pos
            # print('x = ', self.x)
            # print('y = ', self.y)
            # print('center = ', self.center)
            if last_vertex >= 0:
                link = [[temp_name, temp_id], [vertex.robot_name, vertex.id]]
                self.edge.append(Edge(id=self.edge_id, link=link))
            self.edge_id += 1
        else:# matched
            if current_node.robot_name != temp_name or current_node.id != temp_id:#类似于形成自我回环的情况，这里是可以做后端优化的
                link = [[temp_name, temp_id], [current_node.robot_name, current_node.id]] # connect
                self.edge.append(Edge(id=self.edge_id, link=link))
                self.edge_id += 1
        
        return self.vertex_id, current_node, matched_flag
    
    
    def upgradeFrontierPoints(self, vertex_id=-1, type="new", resolution=0.05):
        picked_vertex = None
        picked_vertex_id = 0
        if type == "new":
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
        has_node = 0
        if vertex_id == -1:
            pass
        frontiers = get_frontier_points(picked_vertex.localMap)#返回一系列边界中心
        temp_fd = []
        temp_fp = []
        temp_nd = []
        for front in frontiers:
            front = np.array([front[0], front[1]])
            frontP = np.array([picked_vertex.pose.pose.position.x, picked_vertex.pose.pose.position.y])
            current_pose = frontP
            dis = np.sqrt(np.sum(np.square(front-center))) * resolution
            dis += 4 #?
            if center[0]>=front[0]:#没看懂这部分角度怎么计算的
                if center[1] >=front[1]:#3象限  3象限
                    theta = np.arctan((center[0]-front[0])/(center[1]-front[1]))
                    angle = math.degrees(theta) - 180
                    print('in first quadrant--','origin data is ', center[0],'  ', center[1],'  ',front[0],'  ',front[1],  'changed data is: ', angle)
                else:#2象限 4象限
                    theta = np.arctan((center[0]-front[0])/(front[1]-center[1]))
                    angle = - math.degrees(theta)
                    print('in second quadrant--','origin data is ', center[0],'  ', center[1],'  ',front[0],'  ',front[1],  'changed data is: ', angle)
            else:
                if center[1] >=front[1]:#1象限  2象限
                    theta = np.arctan((center[1]-front[1])/(front[0]-center[0]))
                    angle = 90 + math.degrees(theta)
                    print('in third quadrant--','origin data is ', center[0],'  ', center[1],'  ',front[0],'  ',front[1],  'changed data is: ', angle)
                else:#4象限   1象限
                    theta = np.arctan((front[0]-center[0])/(front[1]-center[1]))
                    angle = math.degrees(theta)
                    print('in fourth quadrant--','origin data is ', center[0],'  ', center[1],'  ',front[0],'  ',front[1],  'changed data is: ', angle)
            odom_angle = math.radians(angle + self.offset_angle) # 后面没仔细看，写的太差了
            map_angle = math.radians(angle)
            front_in_map = copy.deepcopy(frontP)
            frontP[0] += dis * np.cos(odom_angle)
            frontP[1] += dis * np.sin(odom_angle)
            front_in_map[0] += dis * np.cos(map_angle)
            front_in_map[1] += dis * np.sin(map_angle)
            if has_node == 0:
                dis -= 4
                temp_fd.append(dis)
                temp_fp.append(frontP)
                temp_nd.append(angle)
        if len(temp_nd) < len(self.vertex[picked_vertex_id].navigableDirection) and type=="old":
            self.vertex[picked_vertex_id].frontierDistance = temp_fd
            self.vertex[picked_vertex_id].frontierPoints = temp_fp
            self.vertex[picked_vertex_id].navigableDirection = temp_nd
        if type == "new":
            self.vertex[picked_vertex_id].frontierDistance = temp_fd
            self.vertex[picked_vertex_id].frontierPoints = temp_fp
            self.vertex[picked_vertex_id].navigableDirection = temp_nd

        return picked_vertex_id
    
    def plot(self, size, vcolor=(0, 0, 255), ecolor=(0, 255, 0)) -> np.ndarray:
        mapv = np.zeros([size, size, 3], np.uint8)
        pose = dict()
        for vertex in self.vertex:
            pose[vertex.id] = (int((vertex.pose.pose.position.x+5)/10 * size), int((5-vertex.pose.pose.position.y)/10 * size))
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
            