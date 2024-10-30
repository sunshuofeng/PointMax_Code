import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
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



class ScanObjectNNDataLoader(Dataset):
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.num_points=args.num_point
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            '/opt/data3/datasets/scannobject/h5_files/main_split', f'{slit_name}_objectdataset_augmentedrot_scale75.h5')
 

        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)

        

    def __len__(self):
        return self.points.shape[0]

    def _get_item(self, index):
        point_set = self.points[index][:self.num_points]
        label = self.labels[index]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)

