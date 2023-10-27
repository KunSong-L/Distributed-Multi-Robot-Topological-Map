#!/usr/bin/python3.8
import rospy
import cv2
import numpy as np
import time
from robot_function import *


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


    def topo_recon_local_free_space(self,origin_map_shape, origin,resolution=0.05):
        #reconstruction of local free space of multi-FHT-Map
        # origin_map_shape: 2*1
        # origin: origin of the map 2*1
        # resolution: resolution of the map
        #init self.fht_map_multi first

        # 创建一个空白图像，与要填充的矩形大小相同
        #将多个机器人的地图重建到robot 1
        image_shape = origin_map_shape
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
        cv2.imwrite(debug_path+"filled_rectangle.jpg", image)

        return image, new_origin


if __name__ == '__main__':
    time.sleep(3)
    rospy.init_node('multi_robot_expore')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    node = multi_robot_expore(robot_num)

    rospy.spin()