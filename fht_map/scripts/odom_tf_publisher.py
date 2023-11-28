#!/usr/bin/python3.8
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import time
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
# a script with bug dont use it
def odom_publisher():
    time.sleep(3)
    robot_name = rospy.get_param("~robot_name")
    rospy.init_node(robot_name+"_odom_pub")
    pub = rospy.Publisher(robot_name+"/odom", Odometry, queue_size=1)
    rate = rospy.Rate(10) # 10hz
    tf_listener = tf.TransformListener()
    odom = Odometry()
    while not rospy.is_shutdown():
        now_time = rospy.Time.now()
        tf_listener.waitForTransform(robot_name+"/map", robot_name+"/odom", now_time, rospy.Duration(0.1))
        try:
            tf_transform, rotation = tf_listener.lookupTransform(robot_name+"/map", robot_name+"/odom", now_time)
            tf_transform_ready = 1
            odom.header.stamp = now_time
            odom.header.frame_id = robot_name+"/odom"
            odom.pose.pose = Pose(Point(tf_transform[0], tf_transform[1], tf_transform[2]), Quaternion(rotation))
            pub.publish(odom)
        except:
            pass        
        rate.sleep()

if __name__ == '__main__':
    try:
        odom_publisher()
    except rospy.ROSInterruptException:
        pass