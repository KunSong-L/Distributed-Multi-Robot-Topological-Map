#!/usr/bin/python3.8
import rospy
import cv2
import numpy as np
import time
from robot_function import *
from multi_fht_map.srv import frontierSRV, frontierSRVRequest, frontierSRVResponse

debug_path = "/home/master/debug/test1/"
save_result = False

class multi_robot_expore:
    def __init__(self, robot_num):#输入当前机器人name
        self.robot_num =robot_num
        self.robot_list = list()
        for rr in range(robot_num):
            self.robot_list.append("robot"+str(rr+1))
        
        self.fht_map_multi = dict()
        self.free_space_map = None
        self.free_space_origin = [0,0]
        rospy.Service('update_frontier_server', frontierSRV, self.process_frontier_callback)

    def from_local_rect_to_4_point(self,rect,origin,resolution,rotation_mat,angle):
        #rect: [x1,y1,x2,y2]
        point1 = np.array(rect[0:2])
        point2 = np.array(rect[2:])
        #转到像素坐标系下
        point1 = (point1-origin)/resolution
        point1 = point1.astype(np.int16)
        point2 = (point2-origin)/resolution
        point2 = point2.astype(np.int16)
        # 计算矩形的四个顶点坐标
        center = (int((point1[0] + point2[0]) // 2), int((point1[1] + point2[1]) // 2))
        #沿着x方向叫宽，沿着y方向叫高
        tmp = rotation_mat.T @ np.array([point2[0] - point1[0],point2[1] - point1[1],0])
        tmp = tmp.astype(np.int16)
        size = (int(tmp[0]),int(tmp[1]))
        rect = (center, size, angle)
        box = cv2.boxPoints(rect).astype(np.intp)

        return box


    def topo_recon_local_free_space(self,origin_map, origin,resolution=0.05):
        #reconstruction of local free space of multi-FHT-Map
        # origin_map_shape: 2*1
        # origin: origin of the map 2*1
        # resolution: resolution of the map
        #init self.fht_map_multi first

        # 创建一个空白图像，与要填充的矩形大小相同
        #将多个机器人的地图重建到robot 1
        image_shape = origin_map.shape
        origin = np.array(origin)
        #寻找最大最小的坐标
        minU = 0
        maxU = image_shape[1]
        minV = 0
        maxV = image_shape[0]

        for now_robot in self.robot_list:
            if now_robot in self.fht_map_multi.keys():
                nowFhtMap = self.fht_map_multi[now_robot]
            else:
                continue
            if nowFhtMap == None:
                continue
            map_rotation = nowFhtMap.rotation
            angle = int(-np.arctan2(map_rotation[1,0],map_rotation[0,0])/np.pi*180)
            for now_vertex in nowFhtMap.vertex:
                now_local_rect = now_vertex.local_free_space_rect #x1,y1,x2,y2
                if now_local_rect == [0,0,0,0]:
                        continue
                rect_point = self.from_local_rect_to_4_point(now_local_rect,origin,resolution,nowFhtMap.rotation,angle)
                minUV = np.min(rect_point,axis=0)
                maxUV = np.max(rect_point,axis=0)
                #判断是否超过范围
                minU = min(minU,minUV[0]) 
                minV = min(minV,minUV[1]) 
                maxU = max(maxU,maxUV[0]) 
                maxV = max(maxV,maxUV[1])

        #扩充地图
        #left,right,up,down
        padding = [0-minU,maxU-image_shape[1],0-minV,maxV - image_shape[0]]
        new_image_shape = (image_shape[0] + padding[2] + padding[3], image_shape[1] +  padding[0] + padding[1])
        #更新origin
        new_origin = np.array([origin[0] - padding[0]*resolution, origin[1] - padding[2]*resolution])
        image = np.full(new_image_shape,255, dtype=np.uint8) #可能尺寸不够
        image[padding[2]:image_shape[0]+padding[2],padding[0]:image_shape[1]+padding[0]] = copy.deepcopy(origin_map)

        for now_robot in self.robot_list:
            if now_robot in self.fht_map_multi.keys():
                nowFhtMap = self.fht_map_multi[now_robot]
            else:
                continue
            if nowFhtMap == None:
                continue
            map_rotation = nowFhtMap.rotation
            angle = int(-np.arctan2(map_rotation[1,0],map_rotation[0,0])/np.pi*180)
            
            for now_vertex in nowFhtMap.vertex:
                now_local_rect = now_vertex.local_free_space_rect #x1,y1,x2,y2
                if now_local_rect == [0,0,0,0]:
                    continue
                box = self.from_local_rect_to_4_point(now_local_rect,new_origin,resolution,nowFhtMap.rotation,angle)
                temp_image = np.full(new_image_shape,255, dtype=np.uint8) 
                cv2.fillPoly(temp_image, [box], 0)
                image = cv2.bitwise_and(image, temp_image)

        # 显示结果图像
        # cv2.imshow("Filled Rectangle", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(debug_path+"filled_rectangle.jpg", image)

        self.free_space_map = image
        self.free_space_origin = new_origin


    def process_frontier_callback(self,request):
        tmp = np.array(request.frontier)
        frontier_pose = tmp.reshape((-1,2))
        
        now_robot = request.robotname
        #check whether is a explored frontier
        n_point = frontier_pose.shape[0]
        response = frontierSRVResponse()
        if self.free_space_map is None or self.fht_map_multi[now_robot] is None:#没构建free space map或者没匹配上
            response.is_frontier = [True for i in range(n_point)]
            return response
        else:
            #change frontier to robot1 frame
            map_rotation = self.fht_map_multi[now_robot].rotation
            map_trans = self.fht_map_multi[now_robot].trans_vector
            frontier_pose = np.hstack((frontier_pose,np.zeros((n_point,1))))
            changed_pose = (map_rotation @ frontier_pose.T + map_trans.reshape((-1,1))).T
            #以前用的代码，现在不用for循环加速处理了
            # result = []
            # for frontier in changed_pose:
            #     is_explored = self.is_explored_frontier(frontier)
            #     result.append(not is_explored)
            tmp = ((changed_pose[:,0:2]-self.free_space_origin)/0.05).astype(int)
            global_map = self.free_space_map
            on_image_index = np.logical_and.reduce([tmp[:,1]>=0,tmp[:,1]< global_map.shape[0],tmp[:,0]>=0,tmp[:,0]< global_map.shape[1]])
            result = np.logical_not(on_image_index)
            on_image_pose = tmp[on_image_index]
            explored_index = global_map[on_image_pose[:,1],on_image_pose[:,0]]!=0#已经被探索过的下标
            result[on_image_index] = explored_index
            response.is_frontier = result.tolist()
            return response

    def is_explored_frontier(self,pose_in_world):
        #input pose in world frame
        map_res = 0.05
        map_origin = self.free_space_origin
        global_map =  self.free_space_map

        frontier_position = np.array([int((pose_in_world[0] -map_origin[0])/map_res), int((pose_in_world[1] - map_origin[1])/map_res)])
        if frontier_position[1]<0 or frontier_position[1] > global_map.shape[0]-1 or frontier_position[0]<0 or frontier_position[0] > global_map.shape[1]-1:
            return False
        elif global_map[frontier_position[1],frontier_position[0]]==0:
            return True
        else:#没探索
            return False
     
if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('multi_robot_expore')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    node = multi_robot_expore(robot_num)

    rospy.spin()