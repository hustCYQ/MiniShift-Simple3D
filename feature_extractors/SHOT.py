import numpy as np
from sklearn.neighbors import NearestNeighbors
# from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import *
from feature_extractors.pointnet2_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_normals(points, k=10):
    """Compute normal vectors for each point using PCA on k-nearest neighbors."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = []
    
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov_matrix = np.cov(neighbors - np.mean(neighbors, axis=0), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normal = eigvecs[:, np.argmin(eigvals)]
        normals.append(normal)
    
    return np.array(normals)

def compute_shot_descriptor(points, normals, keypoints, radius=0.2, bins=8):
    """Compute SHOT descriptor for given keypoints."""
    descriptors = []
    
    for keypoint in keypoints:
        # 找到 keypoint 最近的法向量
        idx = np.argmin(np.linalg.norm(points - keypoint, axis=1))
        normal = normals[idx]
        
        # 构造局部参考帧 (LRF)
        tangent_x = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(tangent_x) < 1e-6:
            tangent_x = np.cross(normal, [0, 1, 0])
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x)
        
        # 计算邻域点
        nbrs = NearestNeighbors(radius=100,n_neighbors=40).fit(points)
        indices = nbrs.radius_neighbors([keypoint], return_distance=False)[0]
        local_points = points[indices] - keypoint
        local_normals = normals[indices]
        
        # 转换到局部坐标系
        x_vals = np.dot(local_points, tangent_x)
        y_vals = np.dot(local_points, tangent_y)
        z_vals = np.dot(local_points, normal)
        
        # 计算梯度方向 (角度)
        angles = np.arctan2(y_vals, x_vals)  # 计算平面角度
        
        # 计算直方图
        hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
        descriptors.append(hist)
    
    return np.array(descriptors)

def get_SHOT(unorganized_pc):
    unorganized_pc = unorganized_pc.squeeze(0).numpy()
    print(unorganized_pc.shape)

    normals = compute_normals(unorganized_pc)

    # fps the centers out
    # unorganized_pc_no_zeros expected (1,n,3)
    unorganized_pc_no_zeros = unorganized_pc
    unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
    unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)

    batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
    center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), 1024)  # B G 3
    # print(center.shape)
    features = compute_shot_descriptor(unorganized_pc, normals,center.squeeze().cpu().numpy())
    features = torch.tensor(features).cuda()
    agg_point_feature = features.float()
    unorganized_pc = torch.tensor(unorganized_pc)

    # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
    return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center




# 示例使用
if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(100, 3)  # 生成随机点云
    normals = compute_normals(points)
    # keypoints = points[np.random.choice(len(points), 5, replace=False)]  # 随机选取5个关键点
    shot_descriptors = compute_shot_descriptor(points, normals, points)
    print("SHOT Descriptor Shape:", shot_descriptors.shape)