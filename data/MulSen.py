import pathlib
import csv
from torch.utils.data import Dataset
import glob
import os
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader
import re
import csv
from scipy.spatial import KDTree


DATASETS_PATH = '/home/ud202380215/FPFH_3D/data/MulSen_AD'





def mulsen_classes():
    return [
        "capsule",
        "cotton",
        "cube",
        "spring_pad",
        "screw",
        "screen",
        "piggy",
        "nut",
        "flat_pad",
        'plastic_cylinder',
        "zipper",
        "button_cell",
        "toothbrush",
        "solar_panel",
        "light",
    ]



point_th = 40000



def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class DatasetMulSen_ad_train(Dataset):
    def __init__(self, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = DATASETS_PATH
        self.train_sample_list = glob.glob(os.path.join(self.dataset_dir,cls_name, 'Pointcloud', 'train') + "/*.stl")
        self.if_norm = if_norm

    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)

        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points

    def __getitem__(self, idx):

        mesh_stl = o3d.geometry.TriangleMesh()
        mesh_stl = o3d.io.read_triangle_mesh(self.train_sample_list[idx])
        mesh_stl = mesh_stl.remove_duplicated_vertices()
        pc = np.asarray(mesh_stl.vertices)
        N = pc.shape[0]
        pointcloud = self.norm_pcd(pc)


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud) 

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pointcloud) 


        voxel_size_setting = 0.1
        while len(pcd_tmp.points) > point_th:
            pcd_tmp = pcd.voxel_down_sample(voxel_size=voxel_size_setting) 
            voxel_size_setting = voxel_size_setting + 0.1

        pointcloud = np.asarray(pcd_tmp.points)
        

        mask = np.zeros((pointcloud.shape[0]))
        label = 0
        return pointcloud, mask, label, self.train_sample_list[idx]

    def __len__(self):
       return len(self.train_sample_list)



class DatasetMulSen_ad_test(Dataset):
    def __init__(self, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = DATASETS_PATH
        self.if_norm = if_norm
        self.cls_name = cls_name
        self.test_sample_list, self.labels,self.tot_labels = self.load_dataset() 
        self.gt_path = str(os.path.join(self.dataset_dir, self.cls_name, 'Pointcloud','GT'))
    
    def sort(self, file_paths):
        
        paths_with_numbers = []
        pattern = re.compile(r'(\d+)\.(png|stl)$')
        for path in file_paths:
            match = pattern.search(path)
            if match:
                number = int(match.group(1))
                paths_with_numbers.append((path, number))
        paths_with_numbers.sort(key=lambda x: x[1])    
        sorted_paths = [p[0] for p in paths_with_numbers]
        return sorted_paths
    
    
    def load_dataset(self):
        test_sample_list = []
        labels_list = []
        tot_labels_list = []
        defect_types = os.listdir(os.path.join(self.dataset_dir,self.cls_name, 'Pointcloud', 'test'))
        for defect_type in defect_types:
            if defect_type == "good":
                test_good_sample_list = glob.glob(os.path.join(self.dataset_dir,self.cls_name, 'Pointcloud', 'test', defect_type) + "/*.stl")
                test_good_sample_list = self.sort(test_good_sample_list)
                labels_list.extend([0] * len(test_good_sample_list))
                test_sample_list.extend(test_good_sample_list)
                tot_labels_list.extend([[0,0,0] for _ in range(len(test_good_sample_list))])
            else:
                with open(os.path.join(self.dataset_dir,self.cls_name,'RGB','GT',defect_type,'data.csv'),'r') as file:
                    csvreader = csv.reader(file)
                    header = next(csvreader)
 
                    for row in csvreader:
                        object, label1, label2, label3 = row
                    
                        tot_labels_list.append([int(label1),int(label2),int(label3)])
                        if (int(label1)==1) or (int(label2)==1) or (int(label3)==1):
                            label_s = 1
                        else:
                            label_s = 0
                        labels_list.extend([int(label_s)])
                test_nogood_sample_list = glob.glob(os.path.join(self.dataset_dir,self.cls_name, 'Pointcloud', 'test', defect_type) + "/*.stl")
                test_nogood_sample_list= self.sort(test_nogood_sample_list)
                test_sample_list.extend(test_nogood_sample_list)

        return test_sample_list, labels_list,tot_labels_list
            
            


    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)

        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
    
  
    def create_mask(self,pointcloud, pc):
        mask = []
        for point in pc:
            if point in pointcloud:
                mask.append(1)
            else:
                mask.append(0)
        return np.array(mask)

    def mark_stl_with_anomalies(self, stl_vertices, txt_points, tolerance=1000):
        labels = np.zeros(len(stl_vertices), dtype=int)  

       
        tree = KDTree(stl_vertices)


        for txt_point in txt_points:
            dist, idx = tree.query(txt_point)  
            
            if dist < tolerance:  
                labels[idx] = 1

 
        return labels
    def __getitem__(self, idx):
        sample_path = self.test_sample_list[idx]

        if self.tot_labels[idx][2] == 0:
            mesh_stl = o3d.geometry.TriangleMesh()
            mesh_stl = o3d.io.read_triangle_mesh(sample_path)
       
            mesh_stl = mesh_stl.remove_duplicated_vertices()
            pc = np.asarray(mesh_stl.vertices)
            N = pc.shape[0]
            pointcloud = self.norm_pcd(pc)
         

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud) 

            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(pointcloud) 


            voxel_size_setting = 0.1
            while len(pcd_tmp.points) > point_th:
                pcd_tmp = pcd.voxel_down_sample(voxel_size=voxel_size_setting) 
                voxel_size_setting = voxel_size_setting + 0.1

            pointcloud = np.asarray(pcd_tmp.points)


            mask = np.zeros((pointcloud.shape[0]))
            label = 0
        else:
            mesh_stl = o3d.geometry.TriangleMesh()
            mesh_stl = o3d.io.read_triangle_mesh(sample_path)
        
            mesh_stl = mesh_stl.remove_duplicated_vertices()
            pointcloud = np.asarray(mesh_stl.vertices)
            
         
            filename = pathlib.Path(sample_path).stem
            anomaly_type = pathlib.Path(sample_path).parent.stem
            txt_path = os.path.join(self.gt_path,anomaly_type,filename + '.txt')
            pcd = np.genfromtxt(txt_path, delimiter=",")

            pointcloud_mask_part = pcd[:, :3]



            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)

            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(pointcloud) 


            voxel_size_setting = 0.1
            while len(pcd_tmp.points) > point_th:
                pcd_tmp = pcd.voxel_down_sample(voxel_size=voxel_size_setting) 
                voxel_size_setting = voxel_size_setting + 0.1

            pointcloud = np.asarray(pcd_tmp.points)


            mask = self.mark_stl_with_anomalies(pointcloud,pointcloud_mask_part)
            label = 1


       



 
        if(self.if_norm):
            pointcloud = self.norm_pcd(pointcloud)


        return pointcloud, mask, label, sample_path



    def __len__(self):
        return len(self.test_sample_list)



def get_mulsen_loader(split, class_name, img_size=224):
    if split in ['train']:
        dataset = DatasetMulSen_ad_train(cls_name=class_name, num_points=1024,if_norm=True)
    elif split in ['test']:
        dataset = DatasetMulSen_ad_test(cls_name=class_name, num_points=1024,if_norm=True)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader


