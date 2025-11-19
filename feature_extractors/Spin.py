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

def compute_spin_image(points, normals, keypoints, bin_size=10, image_size=20, radius=0.2):
    """Compute Spin Image descriptor for given keypoints."""
    descriptors = []
    
    for keypoint in keypoints:
        # 找到 keypoint 最近的法向量
        idx = np.argmin(np.linalg.norm(points - keypoint, axis=1))
        normal = normals[idx]
        
        # 计算邻域点
        nbrs = NearestNeighbors(radius=100,n_neighbors=40).fit(points)
        indices = nbrs.radius_neighbors([keypoint], return_distance=False)[0]
        local_points = points[indices] - keypoint
        
        # 计算点的投影坐标
        height = np.dot(local_points, normal)  # 计算沿法向量的高度
        radial_dist = np.linalg.norm(local_points - np.outer(height, normal), axis=1)  # 计算径向距离
        
        # 计算 Spin Image 的二维直方图索引
        bin_h = ((height + radius) / (2 * radius) * image_size).astype(int)
        bin_r = (radial_dist / radius * image_size).astype(int)
        
        # 生成 Spin Image 矩阵
        spin_image = np.zeros((image_size, image_size))
        for h, r in zip(bin_h, bin_r):
            if 0 <= h < image_size and 0 <= r < image_size:
                spin_image[h, r] += 1
        
        descriptors.append(spin_image.flatten())  # 展平成向量
    
    return np.array(descriptors)



def get_Spin(unorganized_pc):
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
    features = compute_spin_image(unorganized_pc, normals,center.squeeze().cpu().numpy())
    features = torch.tensor(features).cuda()
    agg_point_feature = features
    unorganized_pc = torch.tensor(unorganized_pc)

    # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
    return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center



# 示例使用
if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(10000, 3)  # 生成随机点云
    normals = compute_normals(points)
    spin_images = compute_spin_image(points, normals, points)
    print("Spin Image Descriptor Shape:", spin_images.shape)