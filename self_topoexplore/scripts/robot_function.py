import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

def set_marker(robot_name, id, pose, color=(0.5, 0, 0.5), action=Marker.ADD):
    now = rospy.Time.now()
    marker_message = Marker()
    marker_message.header.frame_id = robot_name + "/odom"
    marker_message.header.stamp = now
    marker_message.ns = "topological_map"
    marker_message.id = id
    marker_message.type = Marker.SPHERE
    marker_message.action = action
    marker_message.pose.position = pose.pose.position
    marker_message.pose.orientation = pose.pose.orientation
    marker_message.scale.x = 0.3
    marker_message.scale.y = 0.3
    marker_message.scale.z = 0.3
    marker_message.color.a = 1.0
    marker_message.color.r = color[0]
    marker_message.color.g = color[1]
    marker_message.color.b = color[2]

    return marker_message

def set_edge(robot_name, id, poses, type="edge"):
    now = rospy.Time.now()
    path_message = Marker()
    path_message.header.frame_id = robot_name + "/odom"
    path_message.header.stamp = now
    path_message.ns = "topological_map"
    path_message.id = id
    if type=="edge":
        path_message.type = Marker.LINE_STRIP
        path_message.color.a = 1.0
        path_message.color.r = 0.0
        path_message.color.g = 1.0
        path_message.color.b = 0.0
    elif type=="arrow":
        path_message.type = Marker.ARROW
        path_message.color.a = 1.0
        path_message.color.r = 1.0
        path_message.color.g = 0.0
        path_message.color.b = 0.0
    path_message.action = Marker.ADD
    path_message.scale.x = 0.05
    path_message.scale.y = 0.05
    path_message.scale.z = 0.05
    path_message.points.append(poses[0])
    path_message.points.append(poses[1])

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