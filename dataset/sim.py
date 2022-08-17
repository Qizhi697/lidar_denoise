import os
import random
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import scipy.spatial as spatial
from dataset.transforms import Compose, RandomHorizontalFlip, Normalize


class DENSE(data.Dataset):
    """`DENSE LiDAR`_ Dataset.

    Args:
        root (string): Root directory of the ``lidar_2d`` and ``ImageSet`` folder.
        split (string, optional): Select the split to use, ``train``, ``val`` or ``all``
        transform (callable, optional): A function/transform that  takes in distance, reflectivity
            and target tensors and returns a transformed version.
    """

    # TODO: Bu class kismina bir bak

    Class = namedtuple('Class', ['name', 'id', 'color'])

    classes = [
        Class('clear', 0, (0, 0, 0)),
        Class('clear', 1, (100, 0, 0)),
        Class('snow', 2, (255, 0, 0)),
    ]
    pr = [
        Class('Precison', 0, (0, 0, 0)),
        Class('Recall', 100, (0, 0, 142)),
    ]

    def __init__(self, root, split='train', transform=None, k_dis=False):
        self.root = root
        self.split = os.path.join(self.root, '{}'.format(split))
        self.transform = transform
        self.k_dis = k_dis
        self.lidar = []
        self.H = 64
        self.W = 400
        self.size = self.H * self.W
        self.label = []
        if self.k_dis:
            self.data = {'distance': [], 'intensity': [], 'kdistance': []}
            for i, dir in enumerate(['velodyne']):
                self.lidar.append(
                    [os.path.join(r, file) for r, d, f in os.walk(os.path.join(self.split, dir)) for file in f])
            self.lidar[0].sort()
        else:
            self.data = {'distance': [], 'intensity': []}
            for i, dir in enumerate(['velodyne']):
                self.lidar.append(
                    [os.path.join(r, file) for r, d, f in os.walk(os.path.join(self.split, dir)) for file in f])
            self.lidar[0].sort()
        print("Loading Data")
        for i, label_file in enumerate(tqdm(self.lidar[0])):
            points = np.fromfile(self.lidar[0][i], dtype=np.float32).reshape(-1, 5)
            if self.k_dis:
                xyz = points[:, :3]
                kdtree = spatial.cKDTree(xyz, 50)
                dd, _ = kdtree.query(xyz, k=30)
                k_dis = np.mean(dd, axis=1).astype(np.float32)
            distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
            intensity = points[:, 3]
            labels = points[:, -1].astype(int)
            labels[labels != 2] = 1
            points_num = len(labels)
            n = points_num // self.size
            # 64c
            index = np.arange(0, n * self.size)
            if points_num % self.size > self.H:
                end = points_num - points_num % self.H
                res_index = np.arange(end - self.size, end)
                index = np.append(index, res_index)
                index = index.reshape((n + 1, self.W, self.H)).transpose(0, 2, 1).astype(int)
            else:
                index = index.reshape((n, self.W, self.H)).transpose(0, 2, 1).astype(int)

            try:
                labels = list(labels[index])
            except:
                print(0)
            distance = list(distance[index])
            intensity = list(intensity[index])
            self.label.extend(labels)
            self.data['distance'].extend(distance)
            self.data['intensity'].extend(intensity)
            if self.k_dis:
                k_distance = list(k_dis[index])
                self.data['kdistance'].extend(k_distance)

    def __getitem__(self, index):
        # label_dict = {0: 0, 1: 1, 2: 2}
        # label_1 = np.vectorize(label_dict.get)(self.label[index])
        # points_file = self.lidar[0][index]
        # file_id = points_file.split('/')[-1].split('.')[0]
        label_1 = self.label[index]
        label = torch.as_tensor(label_1)
        distance = torch.as_tensor(self.data['distance'][index])
        reflectivity = torch.as_tensor(self.data['intensity'][index])
        if self.k_dis:
            k_distance = torch.as_tensor(self.data['kdistance'][index])
            param_lists = [distance, reflectivity, k_distance]
        else:
            param_lists = [distance, reflectivity]

        if self.transform:
            param_lists, label = self.transform(param_lists, label)
        return param_lists, label

    def __len__(self):
        return len(self.label)

    @staticmethod
    def num_classes():
        return len(DENSE.classes)

    @staticmethod
    def mean():
        # return [24.03, 10.67]
        return [0.21, 12.12]

    @staticmethod
    def std():
        # return [20.35, 16.03]
        return [0.16, 12.32]

    @staticmethod
    def class_weights():
        # return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])
        return torch.tensor([1 / 15.0, 1.0, 10.0])
