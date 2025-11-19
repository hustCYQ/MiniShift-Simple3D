"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from feature_extractors.pointnet2_utils import *
from pointnet2_ops import pointnet2_utils
import cv2
import os
from utils.mvtec3d_util import *
import time
import open3d as o3d
# from feature_extractors.models import *
from torch.utils.data import DataLoader
from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

class Features(torch.nn.Module):


    def unorganized_data_to_organized(self,unorganized_pc, none_zero_data_list):
        '''

        Args:
            unorganized_pc:
            none_zero_data_list:

        Returns:

        '''
        # print(none_zero_data_list[0].shape)
        if not isinstance(none_zero_data_list, list):
            none_zero_data_list = [none_zero_data_list]

        for idx in range(len(none_zero_data_list)):
            none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

        # print("unorganized_pc",unorganized_pc.shape)


        unorganized_pc = unorganized_pc.numpy()
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            

        full_data_list = []

        for none_zero_data in none_zero_data_list:
            if none_zero_data.ndim == 1:
                none_zero_data = np.expand_dims(none_zero_data,1)
            full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
            
            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                full_data[nonzero_indices, :] = none_zero_data
            else:
                full_data = none_zero_data

            full_data_reshaped = full_data.reshape((1, unorganized_pc.shape[0], none_zero_data.shape[1]))
            full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
            full_data_list.append(full_data_tensor)

        return full_data_list

    def normalize(self,pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)


    def apply_ad_scoremap(self,image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)




    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9,args = None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.deep_feature_extractor = Model(device=self.device)
        # self.deep_feature_extractor.to(self.device)
        # self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        self.args = args
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.anomaly_patch_lib = []
        self.pre_patch_lib = []
        self.tmp_patch_lib = []
        self.name_list = []
        self.test_patch_lib = []


        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def init_para(self):
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0




    def compute_anomay_scores(self, patch, mask, label, path, unorganized_pc, unorganized_pc_no_zeros, center):

        dist = torch.cdist(patch, self.patch_lib)  
        min_val, min_idx = torch.min(dist, dim=1)

        feature_map_dims = patch.shape[0]
        s_map = min_val.view(1, 1, feature_map_dims)


        if self.args.use_LFSA:
            s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).to(self.args.device), center.permute(0,2,1).to(self.args.device), s_map.to(self.args.device)).permute(0,2,1)
            s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                s_map = s_map.squeeze().reshape(1,224,224)
                s_map = self.blur(s_map)

            else:
                num_group = 1024
                group_size = 12

                batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
                center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), num_group)  # B G 3

                # knn to get the neighborhood
                knn = KNN(k=group_size, transpose_mode=True)
                _, idx = knn(unorganized_pc_no_zeros, center)  # B G M

                ori_idx = idx
                idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
                
                idx = idx + idx_base
                idx = idx.view(-1)
                neighborhood = s_map.reshape(batch_size * num_points, -1)[idx, :]
                neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()
                agg_s_map = torch.mean(neighborhood,-2).view(1, 1, -1)

                s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).cuda(), center.permute(0,2,1).cuda(), agg_s_map.cuda()).permute(0,2,1)
                s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

        s_map = s_map.squeeze(0)
        s = torch.max(s_map)

        if self.args.dataset == 'real':
            s = torch.mean(s_map)
        if self.args.dataset == 'shapenet':
            tmp_s,_ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)
        if self.args.dataset == 'mulsen' or self.args.dataset == 'minishift' or self.args.dataset == 'quan':
            tmp_s,_ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)     
        

        if self.args.vis_save:
            while isinstance(path,list):
                path = path[0]
            from pathlib import Path
            parts = path.split("data", 1) 
            post_data_path = parts[1].lstrip(os.sep) 
            save_path = "./vis-results/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = normalize(s_map.squeeze())

            scoremap = (scoremap.cpu().numpy() * 255).astype(np.uint8)
            scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
            scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
            unorganized_pc = unorganized_pc.squeeze().cpu()
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)

            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())
            save_path = "./vis-results-GT/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = scoremap.cpu().numpy().astype(np.uint8)
            scoremap[mask.flatten().numpy()==1]=np.array([255,0,0])
            scoremap[mask.flatten().numpy()==0]=np.array([0,0,255])
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)
            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())


        self.image_preds.append(s.cpu().numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.cpu().flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())

        self.predictions.append(s_map.squeeze().detach().cpu().squeeze().numpy())
        self.gts.append(mask.squeeze().detach().cpu().squeeze().numpy())








    def calculate_metrics(self,path=None):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        if not path == None:
            numpy_save = normalize(self.image_preds)
            numpy_save = (numpy_save * 255).astype(np.uint8)
            numpy_save_gt = (self.image_labels*255).astype(np.uint8)[:,0]
            numpy_save = np.append(numpy_save, numpy_save_gt, axis=0)
            np.save(path, numpy_save)


        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        else:
            self.au_pro = 0



    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
        n = len(self.patch_lib)

        self.f_coreset = 0.05
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            
            self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)

               

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)

