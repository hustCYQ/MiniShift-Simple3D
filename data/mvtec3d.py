import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np


DATASETS_PATH = '../Semicore/data/mvtec3ds'


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]



class MVTec3D(Dataset):

    def __init__(self, split, class_name):
        self.cls = class_name
        self.data_path = os.path.join(DATASETS_PATH, self.cls, split)


class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.class_name = class_name
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []
        for name in [self.class_name]:
        # for name in mvtec3d_classes():
            self.data_path = os.path.join(DATASETS_PATH, name, 'train')
            pcd_paths = glob.glob(os.path.join(self.data_path, 'good', 'xyz') + "/*.tiff")
            pcd_paths.sort()
            pcd_tot_paths.extend(pcd_paths)
            tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        organized_pc = read_tiff_organized_pc(pcd_path)
        resized_organized_pc = resize_organized_pc(organized_pc)

        unorganized_pc = organized_pc_to_unorganized_pc(resized_organized_pc.permute(1,2,0))
        return unorganized_pc, label, label, pcd_path


class MVTec3DTest(MVTec3D):
    def __init__(self, class_name):
        super().__init__(split="test", class_name=class_name)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def load_dataset(self):
        pcd_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.data_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                pcd_paths = glob.glob(os.path.join(self.data_path, defect_type, 'xyz') + "/*.tiff")
                pcd_paths.sort()
                pcd_tot_paths.extend(pcd_paths)
                gt_tot_paths.extend([0] * len(pcd_tot_paths))
                tot_labels.extend([0] * len(pcd_tot_paths))
            else:
                pcd_paths = glob.glob(os.path.join(self.data_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.data_path, defect_type, 'gt') + "/*.png")
                pcd_paths.sort()
                gt_paths.sort()
                pcd_tot_paths.extend(pcd_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(pcd_tot_paths))

        assert len(pcd_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return pcd_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]
        organized_pc = read_tiff_organized_pc(pcd_path)
        resized_organized_pc = resize_organized_pc(organized_pc)

        if gt == 0:
            gt = torch.zeros(
                [1, resized_organized_pc.shape[0], resized_organized_pc.shape[1]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        unorganized_pc = organized_pc_to_unorganized_pc(resized_organized_pc.permute(1,2,0))

        return unorganized_pc, gt[:1], label, pcd_path


def get_mvtec_loader(split, class_name):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
