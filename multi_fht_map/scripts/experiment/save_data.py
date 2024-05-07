#!/usr/bin/python3.8
import rospy



# from ransac_icp import ransac_icp
from utils.global_icp import global_icp
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
debug_path = "/home/master/multi_fht_map_data/exp_img/robot2"


class multi_rendezvous_manager():
    def __init__(self):#the maxium robot num
        self.cv_bridge = CvBridge()
        self.count = 0
        self.robotname="robot2"

        rospy.Subscriber("/robot2/color/image_raw", Image, self.save_img, queue_size=1)
        


    def save_img(self,data):
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
        cv2.imwrite(debug_path + "/" + f"self.robotname_{self.count}" + ".png",panoramic_view)
        print(debug_path + "/" + f"self.robotname_{self.count}" + ".png")
        self.count +=1
        



            
if __name__ == '__main__':
    rospy.init_node('multi_robot_explore')
    real_robot_explore_manager = multi_rendezvous_manager()

    rospy.spin()

    