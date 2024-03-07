#!/usr/bin/python3.8
#多个机器人交汇的各种数据分析
from numpy.lib.function_base import _median_dispatcher
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import csv
import tf
import os
import glob
import re
from scipy.spatial.transform import Rotation as R
import copy
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32
import subprocess
import signal
import time

class map_analysis:
    def __init__(self, num_robot) -> None:
       
        self.num_robot = num_robot

        self.tf_listener = tf.TransformListener()

        # 创建CSV文件并写入表头
        self.save_path = path  + method + f'data_n{num_robot}_' + file_index + '.csv'
        self.save_path2 = path  + method + f'time_n{num_robot}_' + file_index + '.csv'
        title_csv = []
        title_csv.append('Timestamp')
        self.receive_time_num = 0
        for now_robot in range(self.num_robot):
            title_csv = title_csv + [f"robot{now_robot+1}_x",f"robot{now_robot+1}_y",f"robot{now_robot+1}_yaw"]
        with open(self.save_path , 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(title_csv)
        
        rospy.Subscriber("/robot1/panoramic", Image, self.write_data_callback, queue_size=1)
        rospy.Subscriber('/rend_time_pub', Float32, self.get_time_call_back, queue_size=100)
    
    def get_time_call_back(self,data):
        get_time = data.data
        write_data = []
        write_data.append(-1)
        write_data = write_data + [0 for i in range(self.num_robot) for j in range(3)]
        write_data.append(get_time)
        with open(self.save_path2, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(write_data)
        self.receive_time_num += 1

        #对于第三次收到get time这个函数，就意味着需要录制rosbag了
        if self.receive_time_num == 3: 
            target_dir = path  + method + f'data_n{num_robot}_' + file_index + '.bag'
            process = subprocess.Popen("rosbag record --duration=10 -O "+ target_dir +" /robot1/topomap /robot2/topomap /robot3/topomap __name:=my_bag", shell=True, env=os.environ) #change to your file path
            time.sleep(11)
            # 发送SIGINT信号给进程，让它结束记录
            process.send_signal(signal.SIGINT)
            process.wait()  # 等待进程结束
            print("----------FHT-Map Record Finished!-----------")
            self.finish_explore = True
    
    def write_data_callback(self, map):

        tmptimenow = rospy.Time.now()
        
        write_pose_data = []
        write_pose_data.append(map.header.stamp.to_sec())
        for robot_index in range(self.num_robot):
            self.tf_listener.waitForTransform("/robot0/map", f"/robot{robot_index+1}"+"/base_footprint", tmptimenow, rospy.Duration(0.5))
            self.tf_transform, self.rotation = self.tf_listener.lookupTransform("/robot0/map", f"/robot{robot_index+1}"+"/base_footprint", tmptimenow,)
            rela_pose = [0,0,0]
            rela_pose[0] = self.tf_transform[0]
            rela_pose[1] = self.tf_transform[1]
            rela_pose[2] = R.from_quat(self.rotation).as_euler('xyz', degrees=False)[2]
            write_pose_data = write_pose_data + rela_pose


        with open(self.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(write_pose_data)
    
if __name__ == '__main__':
    #如果被amcl调用，则同时生成fht map的结果和amcl的结果
    rospy.init_node("map_analysis")

    sim_env = rospy.get_param('~sim_env')
    method = rospy.get_param("~method")
    path = "/home/master/multi_fht_map_data/ComparasionStudy/data/"+ sim_env +"/" + method+"/"
    file_paths = glob.glob(os.path.join(path, "*"))
    
    # 按文件名进行排序
    sorted_file_paths = sorted(file_paths, key=lambda x: os.path.basename(x))

    # 使用正则表达式提取所有数字
    if len(sorted_file_paths)==0:
        file_index = "1"
    else:
        numbers = re.findall(r"\d+", sorted_file_paths[-1])
        numbers = [int(number) for number in numbers]
        file_index = str(numbers[-1] + 1)
    
    num_robot = rospy.get_param("~num_robot")
    
    node = map_analysis(num_robot)
    rospy.spin()