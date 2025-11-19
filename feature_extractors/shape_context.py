import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.mvtec3d_util import *
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

def compute_3d_shape_context(points, bins_r=5, bins_theta=12, bins_phi=8, r_min=0.1, r_max=1.0):
    """Compute 3D Shape Context descriptor for each point."""
    nbrs = NearestNeighbors(n_neighbors=points.shape[0]).fit(points)
    descriptors = []
    log_r = np.logspace(np.log10(r_min), np.log10(r_max), bins_r)
    
    for i, p in enumerate(points):
        distances, indices = nbrs.kneighbors([p], return_distance=True)
        distances = distances.flatten()
        neighbors = points[indices.flatten()[1:]] - p
        
        r = np.linalg.norm(neighbors, axis=1)
        theta = np.arccos(neighbors[:, 2] / (r + 1e-9))  # Elevation angle
        phi = np.arctan2(neighbors[:, 1], neighbors[:, 0])  # Azimuthal angle
        
        hist, _ = np.histogramdd((r, theta, phi), bins=(log_r, bins_theta, bins_phi))
        descriptors.append(hist.flatten())
    
    return np.array(descriptors)

# Example usage

def get_shape_context(unorganized_pc):
    unorganized_pc = unorganized_pc.squeeze(0).numpy()
    print(unorganized_pc.shape)
    features = compute_3d_shape_context(unorganized_pc)
    print(features.shape)

    # fps the centers out
    # unorganized_pc_no_zeros expected (1,n,3)
    unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
    unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)
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
    # print("neighborhood",neighborhood.shape)
    agg_point_feature = torch.mean(neighborhood,-2)

    # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
    return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center

if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(10000, 3)  # Random 3D points
    descriptors = compute_3d_shape_context(points)
    print("3D Shape Context Descriptors:", descriptors.shape)
