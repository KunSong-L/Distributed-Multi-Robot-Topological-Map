import open3d as o3d
import numpy as np
import random
import sys
import copy
from sklearn.decomposition import PCA
import math

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size *4 
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def prepare_pc(pcd, volSize, downSave=False, outlier=False, draw=False, pcaTag=False):
    oldPcd = copy.deepcopy(pcd)
    
    oldNum = np.asarray(oldPcd.points).shape[0]

    if downSave:
        while True:
            volSize *= 1.1
            pcd = oldPcd.voxel_down_sample(voxel_size=volSize)
            tmp = np.asarray(pcd.points).shape[0]
            if  tmp <= min(10000, oldNum-1):
                break
    else:
        pcd = oldPcd.voxel_down_sample(voxel_size=volSize)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamKNN(knn=20))

    return pcd, fpfh

def global_icp(source_pc, target_pc,init_yaw_guess, vis = False):
    voxel_size = 0.05
    source_down, source_fpfh = prepare_pc(source_pc, voxel_size)
    target_down, target_fpfh = prepare_pc(target_pc, voxel_size)
    result_T = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)

    dis_th = voxel_size * 4
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, dis_th, result_T.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
	
    matched_point_number = len(reg_p2p.correspondence_set)

    if vis:
        print(reg_p2p)
        draw_registration_result(source_down, target_down, reg_p2p.transformation)

    if matched_point_number > 300:
        rotation = result_T.transformation[0:3,0:3]
        translation = result_T.transformation[0:3,3]

		#no x,y rotation
        if rotation[2,2] > 0.95:
            return rotation,translation.reshape((3,1))

    return None, None