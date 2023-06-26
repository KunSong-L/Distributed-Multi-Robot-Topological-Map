import random
import copy
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import open3d as o3d
from ransac_icp import *



def ransac(data, fit_model, distance_func, sample_size, max_distance, guess_inliers):
    #fit_model(data): input:data;  output:model
    #distance_func(data,model):return n*1 array of distance
    #
    best_inlier_num = 0
    best_model = None
    max_iterations = int(math.log(1 - 0.9999)/math.log(1 - guess_inliers**sample_size))
    random.seed(0)
    point_num = data.shape[0]
    best_inlier_index = np.ones((point_num,1),dtype=bool)

    goal_inlier_num = point_num*guess_inliers*3
    max_iterations = 2000
    # print("max interation num = ",max_iterations)
    for i in range(max_iterations):
        sub_data = np.array(random.sample(list(data), int(sample_size)))

        now_model = fit_model(sub_data)
        now_dis = distance_func(data,now_model)
        inlier_index = now_dis[:,0]<max_distance[0]
        for i in range(1,len(max_distance)):
            inlier_index = inlier_index * (now_dis[:,i]<max_distance[i])
            
        inlier_num = np.sum(inlier_index)
        
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_model = fitLineFcn(data[inlier_index])
            best_inlier_index = inlier_index
            if inlier_num > goal_inlier_num:
                break

    return best_model, best_inlier_index, i


def remap2T(origin,low,high):
    #将原始值转换到low到high之间
    LH_length = high - low
    result = copy.deepcopy(origin)
    for i in range(0,origin.shape[0]):
        now = origin[i]
        dis = abs(now - low)
        num = int(dis/LH_length)
        if now < low:
            result[i] = now + (num+1)*LH_length
        elif now > high:
            result[i] = now - num*LH_length
    
    return result


def fitLineFcn(points) :
    #给定一系列的point，返回拟合得到的线方程
    #point: n*m array
    length = np.shape(points)[1]
    result = np.zeros((1,length))
    for i in range(0,length):
        now_points = points[:,i]
        if max(now_points) > 160:
            index = now_points>160
            now_points[index] = now_points[index]- 360

        result[0,i] = remap2T(np.array([np.mean(now_points)]),-180,180)
        
    return result

def evalLineFcn(model,points):
    # distance evaluation function
    # return np.sum(np.minimum(abs(points - model),360-abs(points -  model)),1)
    return np.minimum(abs(points - model),360-abs(points -  model))
    

def Fast_Robutst_2PC(pc1,pc2):
    #pc1:2*2pc2:2*x
    #PC1:[x1 y1x2 y2]
    A = np.array([[pc1[0,0]*pc2[0,1], -pc2[0,1]],[pc1[1,0]*pc2[1,1], -pc2[1,1]]])
    B = np.array([[pc2[0,0]*pc1[0,1], pc1[0,1]], [pc2[1,0]*pc1[1,1], pc1[1,1]]])
    C = np.linalg.pinv(B)@A
    tmp = C.T@C

    alpha = tmp[0,0]
    beta = tmp[0,1]
    gamma = tmp[1,1]
    
    # represent the line def
    p = alpha - gamma
    q = 2*beta
    m = alpha + gamma - 2
    #这里和论文给的结果不一样，需要检验一下
    delta2 = p**2 + q**2 - m**2
    if delta2>0:
        delta = delta2**0.5
        #cos(2pi) sin(2pi)
        x_root = [1/(p**2+q**2)*(-p*m + q*delta),1/(p**2+q**2)*(-p*m - q*delta)]
        y_root = [1/(p**2+q**2)*(-q*m - p*delta),1/(p**2+q**2)*(-q*m + p*delta)]
        
        phi = np.array([[np.arctan2(y_root[0],x_root[0])], [np.arctan2(y_root[1],x_root[1])]])/2
        a = np.concatenate((np.cos(phi),np.sin(phi)),axis=1).T
        b = C@a
        theta = np.array([[np.arctan2(b[1,0],b[0,0])], [np.arctan2(b[1,1],b[0,1])]]) + phi
    else:
        return None,None
        tmp = np.array([[np.sign(m)*(-p)/(p**2+q**2)**0.5], [np.sign(m)*(-q)/(p**2+q**2)**0.5]])#cos(2phi) and sin(2phi)
        phi = np.arctan2(tmp[1],tmp[0])[0]/2
        b = C @ np.array([[np.cos(phi)],[np.sin(phi)]])
        theta = np.arctan2(b[1],b[0]) + phi

    return theta,phi

def visualize_matches(img1, kp1, img2, kp2, matches):
    # Draw the matches between the two images
    # The matches argument is a list of DMatch objects, returned by the matcher
    # Create a new image with the two input images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), dtype='uint8')
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    
    # Draw the matches
    for match in matches:
        # Get the keypoints for each image
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        
        # Draw a line between the keypoints in the new image
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, thickness=2)
    
    # Show the new image
    cv.imshow('Matches', vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

def planar_motion_calcu_single(img1,img2,k1,k2,method=1,show_img = 0):
    use_knn_matcher = 1
    if method == 0:
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
    else:
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        if use_knn_matcher:
            bf = cv.BFMatcher()
            total_matches = bf.knnMatch(des1,des2,k=2)
            # Apply ratio test
            matches = []
            for i,(m,n) in enumerate(total_matches):
                if m.distance < 0.5*n.distance:
                    matches.append(m)
        else:
            bf = cv.BFMatcher(crossCheck=True)
            matches = bf.match(des1,des2)
    
    match_point_num = len(matches)
    if match_point_num < 20:
        return [], []

    if show_img:
        visualize_matches(img1, kp1, img2, kp2, matches)
        print("good match num = ", len(matches))      
    
    

    matchedPoints1 = np.ones((3,match_point_num),float)
    matchedPoints2 = np.ones((3,match_point_num),float)

    for i in range(0,match_point_num):
        now_match_point = matches[i]
        matchedPoints1[0:2,i] = np.array(kp1[now_match_point.queryIdx].pt)
        matchedPoints2[0:2,i] = np.array(kp2[now_match_point.trainIdx].pt)

    sort_index = np.argsort(matchedPoints1[0, :])
    matchedPoints1 = matchedPoints1[:,sort_index]
    matchedPoints2 = matchedPoints2[:,sort_index]

    match_point_cam_frame_1 = np.linalg.inv(k1) @ matchedPoints1  
    match_point_cam_frame_2 = np.linalg.inv(k2) @ matchedPoints2  

    # np.save("./data/"+image_file+"/match_point_cam_frame_1.npy",match_point_cam_frame_1)
    # np.save("./data/"+image_file+"/match_point_cam_frame_2.npy",match_point_cam_frame_2)
    total_yaw =np.array([],float)
    total_angleT= np.array([],float)
    #测试过代码正确，就是特征匹配地方效果太差了
    half_len = match_point_num//2
    # calculate relative pose
    for i in range(0,half_len):
        #选择两个
        #按照下标直接选择效果比较差，需要重新选择一下顺序
        #建议这部分做一些误差分析，做一点理论研究
        index_1 = i
        index_2 = i + half_len
        P_i = np.array([[match_point_cam_frame_1[0,index_1],match_point_cam_frame_1[1,index_1]], 
                        [match_point_cam_frame_1[0,index_2],match_point_cam_frame_1[1,index_2]]])
        P_j = np.array([[match_point_cam_frame_2[0,index_1],match_point_cam_frame_2[1,index_1]], 
                        [match_point_cam_frame_2[0,index_2],match_point_cam_frame_2[1,index_2]]])
        
        # P_i = match_point_cam_frame_1[0:2,i:i+2].T
        # P_j = match_point_cam_frame_2[0:2,i:i+2].T

        [Yaw, AngleT] = Fast_Robutst_2PC(P_i,P_j)#theta：旋转角度；phi：位移角度

        try:
            total_yaw = np.append(total_yaw,Yaw.T)
            total_angleT = np.append(total_angleT,AngleT.T)
        except:
            pass


    total_yaw = total_yaw/np.pi*180
    total_angleT = total_angleT/np.pi*180
    

    #return yaw and translation angle in degree
    return total_yaw, total_angleT


def planar_motion_calcu_mulit(img1,img2,k1,k2,cam_pose, pc1 , pc2, show_img=0):
    #cam pose:[[x,y,yaw],...]
    estimated_yaw_min_num = 200
    cam_num = len(cam_pose)
    height = img1.shape[0]
    widht = img1.shape[1]//4
    img1_list = []
    img2_list = []
    for i in range(cam_num):
        img1_tmp = img1[0:height,i*widht:(i+1)*widht]
        img2_tmp = img2[0:height,i*widht:(i+1)*widht]

        img1_list.append(img1_tmp)
        img2_list.append(img2_tmp)

       
    total_yaw =np.array([],float)
    total_angleT= np.array([],float)
    total_ij = np.array([],int).reshape((2,-1))
    
    for i in range(cam_num):
        for j in range(cam_num):
            camera_offset = (cam_pose[i][2] - cam_pose[j][2])/math.pi*180
            image_now_1 = img1_list[i]
            image_now_2 = img2_list[j]

            [Yaw, AngleT] = planar_motion_calcu_single(image_now_1,image_now_2,k1,k2,method=1, show_img = show_img) # use sift FP

            if len(Yaw) == 0:
                continue

            Yaw -= camera_offset 
            total_yaw  = np.concatenate((total_yaw,Yaw))
            total_angleT = np.concatenate((total_angleT,AngleT))

            tmp_ij = np.zeros((2,len(Yaw)),int)
            tmp_ij[0,:] = i
            tmp_ij[1,:] = j
            total_ij  = np.append(total_ij,tmp_ij,axis = 1) #记录了是从第一个坐标系的i到第二个坐标系的j的估计
    
    print("Number of estimated yaw is:", len(total_yaw))
    if len(total_yaw) < estimated_yaw_min_num:
        print("------------Not enough matched feature point!------------------")
        return None
    
    total_yaw = remap2T(total_yaw,-180,180)
    total_angleT = remap2T(total_angleT,-180,180)

    sample_size = 2
    maxDistance = [5]
    yaw_result, best_inlier_index, inter_num = ransac(total_yaw.reshape((-1,1)),fitLineFcn,evalLineFcn,sample_size,maxDistance,guess_inliers = 0.4)

    est_rot = -yaw_result[0][0] 
    if show_img:
        plt.figure()
        x = np.arange(0,total_yaw.shape[0])
        plt.scatter(x,total_yaw,c="b",s=2)
        plt.scatter(x[best_inlier_index],total_yaw[best_inlier_index],c="r",s=2)
        plt.legend(["total","inlier"])
        plt.title("Yaw Angle")
        plt.show()

        plt.figure()
        x = np.arange(0,total_angleT.shape[0])
        plt.scatter(x,total_angleT,c="b",s=2)
        plt.scatter(x[best_inlier_index],total_angleT[best_inlier_index],c="r",s=2)
        plt.legend(["total","inlier"])
        plt.title("Translation Angle")
        plt.show()
        
    print("estimated rot (in degree) is",est_rot)
    # max_correspondence_distance = 0.5  #移动范围的阀值, meter 
    # icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-8, relative_rmse=1e-8)
    # theta = -est_rot/180*np.pi

    # trans_init = np.asarray([[np.cos(theta),-np.sin(theta),0,0],   # 4x4 identity matrix，这是一个转换矩阵，
    #                         [np.sin(theta),np.cos(theta),0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
    #                         [0,0,1,0],   # 这个矩阵为初始变换
    #                         [0,0,0,1]])

    processed_source = o3d.geometry.PointCloud()
    pc2_offset = copy.deepcopy(pc2)
    pc2_offset[2,:] -= 0.1
    processed_source.points = o3d.utility.Vector3dVector(np.vstack([pc2.T,pc2_offset.T]))

    processed_target = o3d.geometry.PointCloud()
    pc1_offset = copy.deepcopy(pc1)
    pc1_offset[2,:] -= 0.1
    processed_target.points = o3d.utility.Vector3dVector(np.vstack([pc1.T,pc1_offset.T]))

    final_R, final_t = ransac_icp(processed_source, processed_target, -est_rot/180*np.pi, vis=show_img)

    if final_R is None or final_t is None:
        return None
    else:
        return [final_t[0][0],final_t[1][0], -math.atan2(final_R[1,0],final_R[0,0])/math.pi*180]
    # #运行icp
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #         processed_source, processed_target, max_correspondence_distance, trans_init,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint(),icp_criteria)

    # trans_1 = copy.deepcopy(reg_p2p.transformation)

    # processed_source.transform(reg_p2p.transformation)

    # #变参数
    # max_correspondence_distance = 0.1  #移动范围的阀值, meter 
    # icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-8, relative_rmse=1e-8)

    # trans_init = np.eye(4,dtype=float)

    # #运行icp
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #         processed_source, processed_target, max_correspondence_distance, trans_init,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint(),icp_criteria)


    # trans_2 = copy.deepcopy(reg_p2p.transformation)
    # final_trans = trans_2 @ trans_1

    # return [final_trans[1,3],final_trans[0,3],-math.atan2(final_trans[1,0],final_trans[0,0])/math.pi * 180 ]


if __name__=="__main__":
    file = "test1/"
    frame_index = "1"
    img1 = cv.imread("/home/master/debug/" + file + "robot1_self" + frame_index +".jpg")
    img2 = cv.imread("/home/master/debug/" + file + "robot1_received" + frame_index +".jpg")
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    x_offset = 0.1
    y_offset = 0.2
    cam_trans = [[x_offset,0,0],[0,y_offset,math.pi/2],[-x_offset,0.0,math.pi],[0,-y_offset,-math.pi/2]]

    # read K
    
    K1_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))
    K2_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))
    
    input_method = 2
    if input_method ==1:
        pc_img1 = cv.imread("/home/master/debug/" + file + "/robot1_local_map.jpg",0)
        pc_img2 = cv.imread("/home/master/debug/" + file + "/robot2_local_map.jpg",0)
        img_width = int(pc_img1.shape[0]/2)
        x1, y1 = np.where((pc_img1 > 90) & (pc_img1 < 110))
        x2, y2 = np.where((pc_img2 > 90) & (pc_img2 < 110))
        resolution = 0.05
        pc1 = np.vstack((x1 - img_width, y1 - img_width, np.zeros(x1.shape,dtype=float))) * resolution
        pc2 = np.vstack((x2 - img_width, y2 - img_width, np.zeros(x2.shape,dtype=float))) * resolution
    else:
    # array input
        loaddata = np.load("/home/master/debug/test1/robot1pc_data" + frame_index +".npz")
        pc1 = loaddata["arr_0"]
        pc2 = loaddata["arr_1"]
    fig = plt.figure(figsize=(8,8))
    plt.scatter(pc1[0,:], pc1[1,:], 6, c='b', marker='o')
    plt.scatter(pc2[0,:], pc2[1,:], 6, c='r', marker='o')
    plt.legend(["robot1","robot2"])
    plt.show()

    pose = planar_motion_calcu_mulit(img1,img2,K1_mat,K2_mat,cam_trans,pc1,pc2,show_img = 1 )

    print(pose)
    
    


    




