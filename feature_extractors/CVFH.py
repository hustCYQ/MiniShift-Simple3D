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







def farthest_point_sampling(points, num_samples):
    """Perform Farthest Point Sampling (FPS) on a set of points.
    
    Args:
        points (numpy.ndarray): Input point cloud of shape (N, 3).
        num_samples (int): Number of points to sample.
    
    Returns:
        numpy.ndarray: Sampled points of shape (num_samples, 3).
    """
    N = points.shape[0]
    sampled_points = np.zeros((num_samples, 3))
    
    # 选择随机的初始点
    idx = np.random.randint(N)
    sampled_points[0] = points[idx]
    
    # 记录所有点到采样集中最近点的距离
    distances = np.full(N, np.inf)
    
    for i in range(1, num_samples):
        # 更新所有点到当前采样集的最小距离
        dist = np.linalg.norm(points - sampled_points[i - 1], axis=1)
        distances = np.minimum(distances, dist)
        
        # 选择最远的点作为下一个采样点
        idx = np.argmax(distances)
        sampled_points[i] = points[idx]
    
    return sampled_points






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

def compute_viewpoint_feature_histogram(points, normals, bins=45):
    """Compute CVFH descriptor based on viewpoint and normal distributions."""
    centroid = np.mean(points, axis=0)
    viewpoint = np.array([0, 0, 0])  # Assume camera at origin
    
    angles = []
    for i in range(points.shape[0]):
        view_vector = viewpoint - points[i]
        view_vector /= np.linalg.norm(view_vector) + 1e-9
        
        angle = np.dot(normals[i], view_vector)
        angles.append(angle)
    
    hist, _ = np.histogram(angles, bins=bins, range=(-1, 1), density=True)
    return hist

def compute_per_point_cvfh(points, normals, bins=45):
    """Compute per-point CVFH descriptor."""
    viewpoint = np.array([0, 0, 0])  # 假设相机位于原点
    descriptors = []

    for i in range(points.shape[0]):
        view_vector = viewpoint - points[i]
        view_vector /= np.linalg.norm(view_vector) + 1e-9
        
        # 计算当前点及其邻域的角度分布
        angles = np.dot(normals, view_vector)
        hist, _ = np.histogram(angles, bins=bins, range=(-1, 1), density=True)
        descriptors.append(hist)

    return np.array(descriptors)

# 计算每个点的特征


def get_CVFH(unorganized_pc):
    unorganized_pc = unorganized_pc.squeeze(0).numpy()
    print(unorganized_pc.shape)

    normals = compute_normals(unorganized_pc)
    features = compute_per_point_cvfh(unorganized_pc, normals)
    features = torch.tensor(features).cuda()
    print(features.shape)

    # fps the centers out
    # unorganized_pc_no_zeros expected (1,n,3)
    unorganized_pc_no_zeros = unorganized_pc
    unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
    unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)
    # unorganized_pc_no_zeros expected (1,n,3)

    batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
    center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), 1024)  # B G 3

    # knn to get the neighborhood
    knn = KNN(k=16, transpose_mode=True)
    _, idx = knn(unorganized_pc_no_zeros, center)  # B G M

    ori_idx = idx
    idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
    
    idx = idx + idx_base
    idx = idx.view(-1)
    neighborhood = features.reshape(batch_size * num_points, -1)[idx, :]
    neighborhood = neighborhood.reshape(batch_size, 1024, 16, -1).contiguous()

    agg_point_feature = torch.mean(neighborhood,-2)
    agg_point_feature = agg_point_feature.squeeze()
    unorganized_pc = torch.tensor(unorganized_pc)

    return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(100, 3)  # Random 3D points
    normals = compute_normals(points)
    per_point_cvfh = compute_per_point_cvfh(points, normals)
    print("CVFH Descriptor:", per_point_cvfh.shape)