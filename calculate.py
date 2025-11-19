from data.mvtec3d import get_mvtec_loader,mvtec3d_classes
from data.real3d import get_real_loader,real3d_classes
from data.anomalyshape import get_shapenet_loader,shapenet3d_classes
from data.MulSen import get_mulsen_loader,mulsen_classes
from data.MiniShiftAD import get_minishift_loader,minishiftAD_classes
import torch
from tqdm import tqdm
from feature_extractors.FPFH import FPFHFeatures
# from feature_extractors.models import *
import numpy as np
import os
from feature_extractors.pointnet2_utils import *


def get_dataloader(dataset_name,split,class_name):
    if dataset_name == 'mvtec':
        return get_mvtec_loader(split, class_name=class_name)
    if dataset_name == 'real':
        return get_real_loader(split, class_name=class_name)
    if dataset_name == 'shapenet':
        return get_shapenet_loader(split, class_name=class_name)
    if dataset_name == 'mulsen':
        return get_mulsen_loader(split, class_name=class_name)
    if dataset_name == 'minishift':
        return get_minishift_loader(split, class_name=class_name) 

    
# 点的总个数
total_points = 0
# 点云的数量
pcd_num = 0
# 需要计算平均点个数与平均异常点百分比
total_anomaly_points = 0


for class_name in mvtec3d_classes():
    test_loader = get_dataloader('mvtec','test',class_name)
    
    for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
        if torch.max(mask) == 0:
            continue
        # print(mask.shape)
        # print(path)
        path = path[0]
        import cv2
        I = cv2.imread(path.replace('xyz','rgb').replace('tiff','png'))
        ratio = I.shape[0]
        mask = torch.flatten(mask)

        pc = pc.squeeze(0).numpy()
        
        nonzero_indices = np.nonzero(np.all(pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = pc[nonzero_indices, :]
        # print(unorganized_pc_no_zeros.shape)
        mask = mask[nonzero_indices]
        # print(mask.shape)

        total_points = total_points + mask.shape[0]*ratio/224.0
        pcd_num = pcd_num + 1
        total_anomaly_points = total_anomaly_points + (mask>0).sum().item()*ratio/224.0

        # print(total_points,pcd_num,total_anomaly_points)
    # break
        



 

#  最后的输出
# 平均每个点云的点数
print("平均每个点云的点数: ",total_points/pcd_num)
# 异常点的百分比
print("异常点的百分比: ",total_anomaly_points/total_points)












