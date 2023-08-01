import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion, PoseStamped, Point
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import copy
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from collections import Counter

height_vertex = 0.5
def set_marker(robot_name, id, pose, color=(0.5, 0, 0.5), action=Marker.ADD, scale = 0.3):
    now = rospy.Time.now()
    marker_message = Marker()
    marker_message.header.frame_id = robot_name + "/odom"
    marker_message.header.stamp = now
    marker_message.ns = "topological_map"
    marker_message.id = id
    marker_message.type = Marker.SPHERE
    marker_message.action = action

    now_vertex_pose = Point()
    now_vertex_pose.x = pose[0]
    now_vertex_pose.y = pose[1]
    now_vertex_pose.z = height_vertex
    now_vertex_ori = Quaternion()
    orientation = R.from_euler('z', pose[2], degrees=True).as_quat()
    now_vertex_ori.x = orientation[0]
    now_vertex_ori.y = orientation[1]
    now_vertex_ori.z = orientation[2]
    now_vertex_ori.w = orientation[3]

    marker_message.pose.position = now_vertex_pose
    marker_message.pose.orientation = now_vertex_ori
    marker_message.scale.x = scale
    marker_message.scale.y = scale
    marker_message.scale.z = scale
    marker_message.color.a = 1.0
    marker_message.color.r = color[0]
    marker_message.color.g = color[1]
    marker_message.color.b = color[2]

    return marker_message

def set_edge(robot_name, id, poses, type="edge", color = (0,1,0), scale = 0.05):
    now = rospy.Time.now()
    path_message = Marker()
    path_message.header.frame_id = robot_name + "/odom"
    path_message.header.stamp = now
    path_message.ns = "topological_map"
    path_message.id = id
    if type=="edge":
        path_message.type = Marker.LINE_STRIP
        path_message.color.a = 1.0
        path_message.color.r = color[0]
        path_message.color.g = color[1]
        path_message.color.b = color[2]
    elif type=="arrow":
        path_message.type = Marker.ARROW
        path_message.color.a = 1.0
        path_message.color.r = 1.0
        path_message.color.g = 0.0
        path_message.color.b = 0.0
    path_message.action = Marker.ADD
    path_message.scale.x = scale
    path_message.scale.y = scale
    path_message.scale.z = scale

    point_1 = Point()
    point_1.x = poses[0][0]
    point_1.y = poses[0][1]
    point_1.z = height_vertex
    path_message.points.append(point_1)

    point_2 = Point()
    point_2.x = poses[1][0]
    point_2.y = poses[1][1]
    point_2.z = height_vertex
    path_message.points.append(point_2)

    path_message.pose.orientation.x=0.0
    path_message.pose.orientation.y=0.0
    path_message.pose.orientation.z=0.0
    path_message.pose.orientation.w=1.0

    return path_message

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

def get_net_param(state):
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    return net_params

def if_frontier(window):
    if 100 in window: # 障碍物
        return False
    if 0 not in window: # 可通过
        return False
    if 255 not in window: # 未知
        return False
    return True

def detect_frontier_old(image):
    kernel_size = 1
    step_size = 3
    frontier_points = []
    shape = image.shape
    for i in range(0,shape[0]-kernel_size,step_size):
        for j in range(0,shape[1]-kernel_size,step_size):
            if if_frontier(image[i - kernel_size : i+kernel_size+1, j - kernel_size : j+kernel_size+1]): #找到已知和未知的边界
                frontier_points.append([i, j])
    
    return np.fliplr(np.array(frontier_points))

def detect_frontier(image):
    kernel = np.ones((3, 3), np.uint8)
    free_space = image ==0
    unknown = (image == 255).astype(np.uint8)
    obs = (image == 100).astype(np.uint8)
    expanded_unknown = cv2.dilate(unknown, kernel).astype(bool)
    expanded_obs = cv2.dilate(obs, kernel).astype(bool)
    near = free_space & expanded_unknown & (~expanded_obs)
    return np.fliplr(np.column_stack(np.where(near)))

def calculate_entropy(array):
    num_bins = 20
    hist, bins = np.histogram(array, bins=num_bins)
    probabilities = hist / len(array)
    probabilities = probabilities[np.where(probabilities != 0)] 
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def sparse_point_cloud(data,delta):
    # 对 xy 平面的前两列进行排序
    data_num = len(data)
    choose_index = np.ones(data_num,dtype=bool)
    check_dick = dict()
    for index, now_point in enumerate(data):
        x = now_point[0]//delta
        y = now_point[1]//delta

        if (x,y) not in check_dick.keys():
            check_dick[(x,y)] = 1
        else:
            choose_index[index] = False
    
    return data[choose_index]

def expand_obstacles(map_data, expand_distance=2):
    map_binary = (map_data == 100).astype(np.uint8)
    kernel_size = 2 * expand_distance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_map_binary = cv2.dilate(map_binary, kernel)
    extended_map = copy.deepcopy(map_data)
    extended_map[expanded_map_binary == 1] = 100

    return extended_map

def find_local_max_rect(image, seed_point, map_origin, map_reso):
    #input: image and (x,y) in pixel frame format start point
    #output rect position of (x1, y1, x2, y2) in pixel frame
    def find_nearest_obstacle_position(image, x, y):
        obstacles = np.argwhere((image == 100) | (image == 255))
        point = np.array([[y, x]])
        distances = cdist(point, obstacles)
        min_distance_idx = np.argmin(distances)
        nearest_obstacle_position = obstacles[min_distance_idx]
        nearest_obstacle_position = np.array([nearest_obstacle_position[1],nearest_obstacle_position[0]])
        return nearest_obstacle_position
    
    vertex_pose_pixel = (np.array(seed_point) - np.array(map_origin))/map_reso
    x = int(vertex_pose_pixel[0])
    y = int(vertex_pose_pixel[1])
    if image[y,x] != 0:
        return [0,0,0,0]
    height, width = image.shape
    nearest_obs_index = find_nearest_obstacle_position(image, x, y)
    

    # 定义左上角和右下角初始值
    if nearest_obs_index[0] < x:
        x1 = nearest_obs_index[0]
        x2 = min(2*x - x1,width)
    else:
        x1 = max(2*x - nearest_obs_index[0],0)
        x2 = nearest_obs_index[0]
    
    if nearest_obs_index[1] < y:
        y1 = nearest_obs_index[1]
        y2 = min(2*y - y1,height)
    else:
        y1 = max(2*y - nearest_obs_index[1],0)
        y2 = nearest_obs_index[1]
    
    if x1 == x:
        y1 += 1
        y2 -= 1
    elif y1 ==y:
        x1 += 1
        x2 -= 1
    else:
        x1 += 1
        y1 += 1
        x1 -= 1
        y2 -= 1
        
    #同时向四个方向扩展
    free_space_flag = [True, True, True, True] #up,left,down,right
    while True in free_space_flag:
        if free_space_flag[0]:
            if y1 < 1 or np.any(image[y1-1, x1:x2+1]):
                free_space_flag[0] = False
            else:
                y1 -= 1
        if free_space_flag[1]:
            if x1 < 1 or np.any(image[y1:y2+1, x1-1]):   
                free_space_flag[1] = False
            else:
                x1 -= 1
        if free_space_flag[2]:
            if y2 > height -2 or np.any(image[y2+1, x1:x2+1]):
                free_space_flag[2] = False
            else:
                y2 += 1
        if free_space_flag[3]:
            if x2 > width -2 or np.any(image[y1:y2+1, x2+1]):
                free_space_flag[3] = False
            else:
                x2 += 1
    x1 = x1 * map_reso + map_origin[0]
    x2 = x2 * map_reso + map_origin[0]
    y1 = y1 * map_reso + map_origin[1]
    y2 = y2 * map_reso + map_origin[1]
    return [x1,y1,x2,y2]

def calculate_vertex_info(frontiers, cluser_eps=1, cluster_min_samples=5):
    # input: frontier; DBSCAN eps; DBSCAN min samples
    # output: how many vertex in this cluster
    dbscan = DBSCAN(eps=cluser_eps, min_samples=cluster_min_samples)
    labels = dbscan.fit_predict(frontiers)
    label_counts = Counter(labels)
    label_counts[-1] = 0
    vertex_infor = [label_counts[now_label] for now_label in labels]

    return vertex_infor