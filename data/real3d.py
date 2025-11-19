import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
DATASETS_PATH = '../Semicore/data/real3d_train_cut'
def real3d_classes():
    return [
        "airplane",   
        "candybar",    
        "car",         
        "chicken",     
        "diamond",      
        "duck",         
        "fish",        
        "gemstone",
        "seahorse",
        "shell",
        "starfish",
        "toffees",
    ]

voxel_size_setting = 0.15
class Real3D(Dataset):

    def __init__(self, split, class_name):
        self.cls = class_name
        self.data_path = os.path.join(DATASETS_PATH, self.cls, split)


class Real3DTrain(Real3D):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        name = self.cls
        self.pcd_path = os.path.join(DATASETS_PATH, name, 'train_cut')
        pcd_paths = glob.glob(self.pcd_path+ "/*.asc")
        pcd_paths.sort()
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        input_points = np.loadtxt(pcd_path,dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_points[:,0:3]) #点云数据
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting) 

        resized_organized_pc = np.asarray(pcd.points)
        unorganized_pc = resized_organized_pc


        return unorganized_pc, label, label, pcd_path


class Real3DTest(Real3D):
    def __init__(self, class_name):
        super().__init__(split="test", class_name=class_name)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.data_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                pcd_paths = glob.glob(os.path.join(self.data_path, defect_type) + "/*.txt")
                pcd_paths.sort()
                pcd_tot_paths.extend(pcd_paths)
                gt_tot_paths.extend([0] * len(pcd_paths))
                tot_labels.extend([0] * len(pcd_paths))
            else:
                pcd_paths = glob.glob(os.path.join(self.data_path, defect_type) + "/*.txt")
                gt_paths = glob.glob(os.path.join(self.data_path, defect_type) + "/*.txt")
                pcd_paths.sort()
                gt_paths.sort()


                pcd_tot_paths.extend(pcd_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(pcd_paths))

        assert len(pcd_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return pcd_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]

        input_points = np.loadtxt(pcd_path,dtype=np.float32)

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        idx1 = input_points[:,3]==0
        idx2 = input_points[:,3]==1
        pcd1.points = o3d.utility.Vector3dVector(input_points[idx1,0:3]) 
        pcd2.points = o3d.utility.Vector3dVector(input_points[idx2,0:3]) 

        pcd = pcd1 + pcd2
        pcd_new = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
        
        pcd1_new = o3d.geometry.PointCloud()
        pcd2_new = o3d.geometry.PointCloud()

        pcd1_vec = []
        pcd2_vec = []
        from scipy.spatial.distance import cdist

        pcd_points_new = np.asarray(pcd_new.points)
        pcd_points = np.asarray(pcd.points)


        pc_len = len(pcd1.points)
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for x in pcd_points_new:
            [k, idx_, _] = pcd_tree.search_knn_vector_3d(x, 1)
            # print(idx_[0])
            if idx_[0] < pc_len:
                pcd1_vec.append(x)
            else:
                pcd2_vec.append(x)

        pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1_vec))
        if not len(pcd2_vec)==0:
            pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd2_vec))

       

        pcd = pcd1 + pcd2

        resized_organized_pc = np.asarray(pcd.points)
        unorganized_pc = resized_organized_pc


        if gt == 0:
            gt = torch.zeros(
                [1, unorganized_pc.shape[0]])
        else:
            gt = torch.zeros(unorganized_pc.shape[0])
            gt[len(pcd1.points):len(pcd.points)] = 1.0
            gt = torch.where(gt > 0.5, 1., .0)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
        return unorganized_pc, gt[:1], label, pcd_path


def get_real_loader(split, class_name):
    if split in ['train']:
        dataset = Real3DTrain(class_name=class_name)
    elif split in ['test']:
        dataset = Real3DTest(class_name=class_name)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader

