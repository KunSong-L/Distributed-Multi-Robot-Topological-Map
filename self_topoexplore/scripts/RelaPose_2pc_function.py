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
from sklearn.cluster import DBSCAN
from scipy import stats

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
        tmp = np.array([[np.sign(m)*(-p)/(p**2+q**2)**0.5], [np.sign(m)*(-q)/(p**2+q**2)**0.5]])#cos(2phi) and sin(2phi)
        phi = np.arctan2(tmp[1],tmp[0])[0]/2
        b = C @ np.array([[np.cos(phi)],[np.sin(phi)]])
        theta = np.arctan2(b[1],b[0]) + phi

    return theta,phi

def planar_motion_calcu_single(img1,img2,k1,k2,method=1):
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
                if m.distance < 0.7*n.distance:
                    matches.append(m)
        else:
            bf = cv.BFMatcher(crossCheck=True)
            matches = bf.match(des1,des2)

    # print("good match num = ", len(matches))      
    match_point_num = len(matches)

    if match_point_num < 10:
        return [], []

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

        total_yaw = np.append(total_yaw,Yaw.T)
        total_angleT = np.append(total_angleT,AngleT.T)


    total_yaw = total_yaw/np.pi*180
    total_angleT = total_angleT/np.pi*180
    

    #return yaw and translation angle in degree
    return total_yaw, total_angleT


def planar_motion_calcu_mulit(img1,img2,k1,k2,cam_pose):
    #cam pose:[[x,y,yaw],...]
    show_img = 1
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
        
    cam_R = []
    cam_t = []
    for pose in cam_pose:
        cam_R.append(R.from_euler('z',pose[2],degrees=False).as_matrix())
        cam_t.append(np.array([[pose[0]],[pose[1]],[0]]))
       
    total_yaw =np.array([],float)
    total_angleT= np.array([],float)
    total_ij = np.array([],int).reshape((2,-1))

    for i in range(cam_num):
        for j in range(cam_num):
            camera_offset = (cam_pose[i][2] - cam_pose[j][2])/math.pi*180
            image_now_1 = img1_list[i]
            image_now_2 = img2_list[j]

            [Yaw, AngleT] = planar_motion_calcu_single(image_now_1,image_now_2,k1,k2,method=1) # use sift FP

            if len(Yaw) == 0:
                continue
            Yaw -= camera_offset
            total_yaw  = np.concatenate((total_yaw,Yaw))
            total_angleT = np.concatenate((total_angleT,AngleT))

            tmp_ij = np.zeros((2,len(Yaw)),int)
            tmp_ij[0,:] = i
            tmp_ij[1,:] = j
            total_ij  = np.append(total_ij,tmp_ij,axis = 1)
    
    
    total_yaw = remap2T(total_yaw,-180,180)
    total_angleT = remap2T(total_angleT,-180,180)

    sample_size = 3
    maxDistance = [5]
    yaw_result, best_inlier_index, inter_num = ransac(total_yaw.reshape((-1,1)),fitLineFcn,evalLineFcn,sample_size,maxDistance,guess_inliers = 0.4)

    t1 = []
    t2 = []
    rho_index = []

    R_robot = R.from_euler('z', -yaw_result, degrees=True).as_matrix().reshape(3,3)
    count = 0
    now_i = total_ij[0,0]
    now_j = total_ij[1,0]
    for i in range(len(best_inlier_index)):
        if best_inlier_index[i] != True:
            continue
        R_i = cam_R[total_ij[0,i]]
        t_i = cam_t[total_ij[0,i]]
        R_j = cam_R[total_ij[1,i]]
        t_j = cam_t[total_ij[1,i]]
        tmp_angle = total_angleT[i]
        now_phi = total_angleT[i]/180*math.pi
        t_cam = np.array([[np.cos(now_phi)], [-np.sin(now_phi)], [0]])

        t1.append(R_i@t_cam)
        t2.append(t_i - R_robot @ t_j)

        if now_i != total_ij[0,i] or now_j != total_ij[1,i]:
            now_i = total_ij[0,i]
            now_j = total_ij[1,i]
            count += 1
        rho_index.append(count)

    if show_img:
        index  = np.arange(0,len(best_inlier_index))
        plt.scatter(index[best_inlier_index],total_yaw[best_inlier_index],s=2)
        plt.scatter(index[np.logical_not(best_inlier_index)],total_yaw[np.logical_not(best_inlier_index)],c = 'r',s=2)
        plt.title("yaw inlier and outlier")
        plt.show()

    #return yaw and translation angle in degree
    return yaw_result,t1,t2,rho_index

if __name__=="__main__":
    img1 = cv.imread("/home/master/debug/robot1.jpg")
    img2 = cv.imread("/home/master/debug/robot2.jpg")
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    x_offset = 0.1
    y_offset = 0.2
    cam_trans = [[x_offset,0,0],[0,y_offset,math.pi/2],[-x_offset,0.0,math.pi],[0,-y_offset,-math.pi/2]]
    cam_R = []
    cam_t = []
    for pose in cam_trans:
        cam_R.append(R.from_euler('z',pose[2],degrees=False).as_matrix())
        cam_t.append(np.array([[pose[0]],[pose[1]],[0]]))

    # read K
    K1_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))
    K2_mat=np.array([319.9988245765257, 0.0, 320.5, 0.0, 319.9988245765257, 240.5, 0.0, 0.0, 1.0]).reshape((3,3))

    cam_index = [1,2,1,1]
    # cam_index = [1,2,2,2]
    best_model,t1,t2,rho_index = planar_motion_calcu_mulit(img1,img2,K1_mat,K2_mat,cam_trans)
    print(best_model[0][0])
    plt.figure()
    rho = np.arange(-2,2,0.1)
    x_plot = np.ones(len(rho),float)
    y_plot = np.ones(len(rho),float)
    for j in range(0,len(t1)):
        for i in range(len(rho)):
            now_t1 = t1[j][0:2]
            now_t2 = t2[j][0:2]
            tmp = rho[i] * now_t1 + now_t2
            x_plot[i] = tmp[0,0]
            y_plot[i] = tmp[1,0]
        plt.plot(x_plot,y_plot)
    plt.xlim((-1,1))
    plt.ylim((-2,2))

    #direction solve
    interextion_line = np.array([],float).reshape((2,-1))
    for i in range(0,len(t1)):
        for j in range(i,len(t1)):
            if rho_index[i]==rho_index[j]:
                continue
            first_t1 = t1[i][0:2]
            first_t2 = t2[i][0:2]

            second_t1 = t1[j][0:2]
            second_t2 = t2[j][0:2]

            e1 = first_t1
            e2 = -second_t1

            #三个向量都在平面上
            A = np.array([[e1[0][0],e2[0][0]],[e1[1][0], e2[1][0]]])
            b = (second_t2 - first_t2)[0:2]
            try:
                res = np.linalg.inv(A) @ b
                rho1 = res[0][0]
                rho2 = res[1][0]
                if abs(rho1)>10 or abs(rho2)>10:
                    continue
                res = rho1*first_t1 + first_t2
                interextion_line = np.append(interextion_line,res,axis = 1)
            except:
                continue
    
    # cluster
    dbscan = DBSCAN(eps=0.02, min_samples=20).fit(interextion_line.T)#聚类
    point_lable = dbscan.labels_
    lables = np.unique(dbscan.labels_)# 获取有几类
    mode, count = stats.mode(dbscan.labels_)
    if mode[0]==-1:
        most_common_label = mode[1]
        num_most_common = count[1]
    else:
        most_common_label = mode[0]
        num_most_common = count[0]
    print("most common label is:  ", most_common_label)
    print("number of common label is:  ", num_most_common)
    points_list = [np.array([],float).reshape((2,-1)) for i in range(len(lables))]#获取每一类具体有多少点

    for i in range(interextion_line.shape[1]):
        points_list[dbscan.labels_[i]] = np.append(points_list[dbscan.labels_[i]],interextion_line[:,i].reshape(2,-1),axis = 1) #把每个点加进对应类里面去
    
    plt.figure()
    plt.scatter(points_list[most_common_label][0,:],points_list[most_common_label][1,:],s=1)

    plt.figure()
    for i in range(len(lables)):
        plt.scatter(points_list[i][0,:],points_list[i][1,:],s=1)
    plt.show()
    print("estimated translation is: x = ",np.mean(points_list[2][0,:]),"   y= ",np.mean(points_list[2][1,:]))



    




