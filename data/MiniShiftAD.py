import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
DATASETS_PATH = '/home/ud202380215/FPFH_3D/data/MiniShiftAD'

def minishiftAD_classes():
    return [
        "capsule",
        "cube",
        "spring_pad",
        "screw",
        "screen",
        "piggy",
        "nut",
        "flat_pad",
        'plastic_cylinder',
        "button_cell",
        "toothbrush",
        "light",
    ]

class MiniShiftAD(Dataset):

    def __init__(self, split, class_name, level='ALL'):
        self.cls = class_name
        self.data_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.level = level


class MiniShiftADTrain(MiniShiftAD):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        name = self.cls
        self.pcd_path = os.path.join(DATASETS_PATH, name, 'train','good')
        pcd_paths = glob.glob(self.pcd_path+ "/*.txt")
        pcd_paths.sort()
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        unorganized_pc = np.loadtxt(pcd_path,dtype=np.float32)
        unorganized_pc[:,0] = unorganized_pc[:,0] - np.mean(unorganized_pc[:,0])
        unorganized_pc[:,1] = unorganized_pc[:,1] - np.mean(unorganized_pc[:,1])
        unorganized_pc[:,2] = unorganized_pc[:,2] - np.mean(unorganized_pc[:,2])



        return unorganized_pc, label, label, pcd_path


class MiniShiftADTest(MiniShiftAD):
    def __init__(self, class_name, level='ALL'):
        super().__init__(split="test", class_name=class_name, level=level)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

        
    def load_dataset(self):
        pcd_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.data_path)
        level = self.level

        for defect_type in defect_types:
            print(defect_type)
            if defect_type == 'good':
                pcd_paths = glob.glob(os.path.join(self.data_path, defect_type) + "/*.txt")
                pcd_paths.sort()
                pcd_tot_paths.extend(pcd_paths)
                gt_tot_paths.extend([0] * len(pcd_paths))
                tot_labels.extend([0] * len(pcd_paths))
            else:
                if level == 'ALL' or level == 'easy':
                    pcd_paths = glob.glob(os.path.join(self.data_path, defect_type,'easy') + "/*.txt")
                    gt_paths = [x.replace('test','gt') for x in pcd_paths]
                    pcd_paths.sort()
                    gt_paths.sort()
                    pcd_tot_paths.extend(pcd_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(pcd_paths))

                if level == 'ALL' or level == 'medium':
                    pcd_paths = glob.glob(os.path.join(self.data_path, defect_type,'medium') + "/*.txt")
                    gt_paths = [x.replace('test','gt') for x in pcd_paths]
                    pcd_paths.sort()
                    gt_paths.sort()
                    pcd_tot_paths.extend(pcd_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(pcd_paths))

                if level == 'ALL' or level == 'hard':
                    pcd_paths = glob.glob(os.path.join(self.data_path, defect_type,'hard') + "/*.txt")
                    gt_paths = [x.replace('test','gt') for x in pcd_paths]
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

        unorganized_pc = np.loadtxt(pcd_path,dtype=np.float32)

        if gt == 0:
            gt = torch.zeros(
                [1, unorganized_pc.shape[0]])
        else:
            gt = np.loadtxt(gt)
            gt = torch.tensor(gt).squeeze()
            gt = torch.where(gt > 0.5, 1., .0)
            gt = gt.unsqueeze(0)

        unorganized_pc[:,0] = unorganized_pc[:,0] - np.mean(unorganized_pc[:,0])
        unorganized_pc[:,1] = unorganized_pc[:,1] - np.mean(unorganized_pc[:,1])
        unorganized_pc[:,2] = unorganized_pc[:,2] - np.mean(unorganized_pc[:,2])
        return unorganized_pc, gt[:1], label, pcd_path


def get_minishift_loader(split, class_name, level='ALL'):
    if split in ['train']:
        dataset = MiniShiftADTrain(class_name=class_name)
    elif split in ['test']:
        dataset = MiniShiftADTest(class_name=class_name,level=level)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader


