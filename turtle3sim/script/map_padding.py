#!/usr/bin/python3.8
from numpy.lib.function_base import _median_dispatcher
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import csv

path = "/home/master/debug/map_complete_data/"
class MapPadding:
    def __init__(self, robot_name) -> None:
        print(robot_name)
        self.self_robot_name = robot_name
        self.map_pub = rospy.Publisher(
            robot_name+"/map", OccupancyGrid, queue_size=10)
        # self.pose_pub = rospy.Publisher(
        #     robot_name+"/testpose", PoseStamped, queue_size=10)
        self.map_timestamps = []
        self.zeros_counts = []
        self.single_robot = 1
        if self.single_robot:
            # 创建CSV文件并写入表头
            with open(path + robot_name + 'map_complete.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Zeros Count'])


        rospy.Subscriber(
            robot_name+"/map_origin", OccupancyGrid, self.map_callback, queue_size=1)
    
    def map_callback(self, map):
        # print(map.info.origin.position)
        map_message = OccupancyGrid()
        map_message.header = map.header
        map_message.info = map.info
        # print("map orientation::", map.info.origin)
        padding = 200
        shape = (map.info.height, map.info.width)
        mapdata = np.asarray(map.data).reshape(shape)
        if self.single_robot:
            # Count the number of zeros in the map
            zeros_count = np.sum(mapdata == 0)
            # Save the map timestamp and number of zeros in a file
            map_time = map.header.stamp.to_sec()
            self.map_timestamps.append(map_time)
            self.zeros_counts.append(zeros_count)
            with open(path + robot_name + 'map_complete.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([map_time, zeros_count])

        localMap = np.full((shape[0]+padding*2, shape[1]+padding*2), -1).astype(np.int8)
        localMap[padding:shape[0]+padding, padding:shape[1]+padding] = mapdata

        map_message.data = tuple(localMap.flatten())
        map_message.info.height += padding*2
        map_message.info.width += padding*2
        map_message.info.origin.position.x -= padding*map.info.resolution
        map_message.info.origin.position.y -= padding*map.info.resolution
        self.map_pub.publish(map_message)
        # before_send = np.asarray(map_message.data).reshape((map_message.info.height, map_message.info.width))

        # pose = PoseStamped()
        # pose.header.frame_id = map_message.header.frame_id
        # pose.pose.position = map_message.info.origin.position
        # pose.pose.orientation.z = 0
        # pose.pose.orientation.w = 1
        # self.pose_pub.publish(pose)


if __name__ == '__main__':
    rospy.init_node("map_padding")
    robot_name = rospy.get_param("~robot_name")
    node = MapPadding(robot_name)
    rospy.spin()