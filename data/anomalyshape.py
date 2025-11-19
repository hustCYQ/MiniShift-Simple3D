import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d

DATASETS_PATH = '/data2/code/Semicore/data/AnomalyShapenet/pcd'

def shapenet3d_classes():
    return [
        "ashtray0",    
        "bag0",          
        "bottle0",
        "bottle1",
        "bottle3",
        "bowl0",
        "bowl1",
        "bowl2",
        "bowl3",
        "bowl4",
        "bowl5",
        "bucket0",
        "bucket1",
        "cap0",
        "cap3",
        "cap4",
        "cap5",
        "cup0",
        "cup1",
        "eraser0",
        "headset0",
        "headset1",
        "helmet0",
        "helmet1",
        "helmet2",
        "helmet3",
        "jar0",
        "microphone0",
        "shelf0",
        "tap0",
        "tap1",
        "vase0",
        "vase1",
        "vase2",
        "vase3",
        "vase4",
        "vase5",
        "vase7",
        "vase8",
        "vase9",

    ]

class Shape3D(Dataset):

    def __init__(self, split, class_name):
        self.cls = class_name
        self.pcd_path = os.path.join(DATASETS_PATH, self.cls, split)


class Shape3DTrain(Shape3D):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []
        name = self.cls
        self.pcd_path = os.path.join(DATASETS_PATH, name, 'train')
        pcd_paths = glob.glob(self.pcd_path+ "/*.pcd")
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        pcd = o3d.io.read_point_cloud(pcd_path)

        unorganized_pc = np.asarray(pcd.points)
        return unorganized_pc, label, label, pcd_path


class Shape3DTest(Shape3D):
    def __init__(self, class_name):
        super().__init__(split="test", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        pcd_paths = glob.glob(self.pcd_path+ "/*.pcd")
        gt_path = glob.glob(self.pcd_path.replace("test",'GT')+ "/*.txt")



        for path in pcd_paths:
            if not "positive" in path:
                continue
            pcd_tot_paths.append(path)
        tot_labels.extend([0]*len(pcd_tot_paths))
        pcd_tot_paths.extend(gt_path)
        tot_labels.extend([1]*len(gt_path))


        assert len(pcd_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]


        if label == 0:
            pcd = o3d.io.read_point_cloud(pcd_path)
            unorganized_pc = np.asarray(pcd.points)
            gt = torch.zeros(
                [1, unorganized_pc.shape[0]])
        else:
            input_points = np.loadtxt(pcd_path,dtype=np.float32,delimiter=',')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(input_points[:,0:3])
            unorganized_pc = np.asarray(pcd.points)

            gt = torch.Tensor(input_points[:,3])


            gt = torch.where(gt > 0.5, 1., .0)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)

        return unorganized_pc, gt[:1], label, pcd_path


def get_shapenet_loader(split, class_name):
    if split in ['train']:
        dataset = Shape3DTrain(class_name=class_name)
    elif split in ['test']:
        dataset = Shape3DTest(class_name=class_name)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader


