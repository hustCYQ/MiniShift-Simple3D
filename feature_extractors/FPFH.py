from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import *
# from feature_extractors.models import *
from feature_extractors.pointnet2_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm


only_use_points = True


from data.real3d import voxel_size_setting
from feature_extractors.shape_context import get_shape_context 
from feature_extractors.CVFH import get_CVFH 
from feature_extractors.NARF import get_NARF 
from feature_extractors.Spin import get_Spin 
from feature_extractors.Unique_shape import get_USC 
from feature_extractors.SHOT import get_SHOT

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'






def batched_knn(knn,reference, query, batch_size=4000):
    all_idx = []
    for i in range(0, query.shape[1], batch_size):
        q_batch = query[:, i:i+batch_size, :]  # shape [B, b, 3]
        _, idx = knn(reference, q_batch)    # shape [B, b, k]
        all_idx.append(idx)
    return torch.cat(all_idx, dim=1)           # [B, G, k]





class FPFHFeatures(Features):

    def __init__(self,args=None):
        self.args = args
        super().__init__(args = args)


    def get_fpfh_features(self,unorganized_pc, voxel_size=0.1):
        # unorganized_pc:(1,n,3)--> (n,3)
        unorganized_pc = unorganized_pc.squeeze(0).numpy() #(n,3)


        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        else:
            unorganized_pc_no_zeros = unorganized_pc
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

        voxel_size = 1000000
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10000, max_nn=10))

        radius_feature = 1000000
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=self.args.max_nn))
        fpfh_np = np.asarray(pcd_fpfh.data.T, dtype=np.float32)
        fpfh = torch.from_numpy(fpfh_np).to(self.args.device)


        if self.args.use_MSND:
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
            (radius=radius_feature, max_nn=2*self.args.max_nn))
            fpfh2 = torch.Tensor(pcd_fpfh.data.T).cuda()
            fpfh = torch.cat([fpfh,fpfh2],dim=-1)


            if self.args.num_MSND == 2:
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
                (radius=radius_feature, max_nn=3*self.args.max_nn))
                fpfh3 = torch.Tensor(pcd_fpfh.data.T).cuda()


                fpfh = torch.cat([fpfh,fpfh2,fpfh3],dim=-1)
        
        # fps the centers out
        # unorganized_pc_no_zeros expected (1,n,3)
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
        unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)
        # unorganized_pc_no_zeros expected (1,n,3)
        batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
        center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), self.args.num_group)  # B G 3

        # knn to get the neighborhood
        knn = KNN(k=self.args.group_size, transpose_mode=True)

        idx = batched_knn(knn,unorganized_pc_no_zeros, center)  # B G M

        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
        
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = fpfh.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.args.num_group, self.args.group_size, -1).contiguous()
        # print("neighborhood",neighborhood.shape)
        agg_point_feature = torch.mean(neighborhood,-2)

        if self.args.use_LFSA:
            agg_point_feature = agg_point_feature.squeeze()
        else:
            agg_point_feature = fpfh.squeeze()

        unorganized_pc = torch.tensor(unorganized_pc)

        # agg_point_feature (group,f)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)

        # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
        return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center

    def get_features(self,unorganized_pc):
        if self.args.feature == 'FPFH':
            return self.get_fpfh_features(unorganized_pc)
        if self.args.feature == 'shape_context':
            return get_shape_context(unorganized_pc)
        if self.args.feature == 'CVFH':
            return get_CVFH(unorganized_pc)
        if self.args.feature == 'NARF':
            return get_NARF(unorganized_pc)
        if self.args.feature == 'Spin':
            return get_Spin(unorganized_pc)
        if self.args.feature == 'USC':
            return get_USC(unorganized_pc)
        if self.args.feature == 'SHOT':
            return get_SHOT(unorganized_pc)  





    def collect_features(self,pc):
        # pc:(1,n,3)

        # agg_point_feature (group,33)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)

        feature_maps,_,_,_ = self.get_features(pc)
        self.patch_lib.append(feature_maps)

    def predict(self, pc, mask, label, path=None):
        # agg_point_feature (group,33)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)

        agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center = self.get_features(pc)
        self.compute_anomay_scores(agg_point_feature, mask, label,path, unorganized_pc,unorganized_pc_no_zeros,center)



 