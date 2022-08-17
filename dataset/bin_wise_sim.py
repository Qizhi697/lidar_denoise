import os
import random
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import h5py
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
        Class('nolabel', 0, (0, 0, 0)),
        Class('clear', 100, (0, 0, 142)),
        Class('snow1', 101, (220, 20, 60)),
    ]

    def __init__(self, root, split='train', transform=None, k_dis=False):
        # self.root = os.path.expanduser(root)
        self.root = root
        self.split = os.path.join(self.root, '{}'.format(split))
        self.transform = transform
        self.k_dis = k_dis
        self.lidar = []
        self.H = 64
        self.W = 50
        self.size = self.H * self.W
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

    def __getitem__(self, index):
        points_file = self.lidar[0][index]
        file_id = points_file.split('/')[-1].split('.')[0]
        # labels_file = os.path.join(self.split, 'labels', file_id + '.label')
        points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 5)
        if self.k_dis:
            # k_dis = np.fromfile(self.lidar[2][index], dtype=np.float32).reshape(-1)
            xyz = points[:, :3]
            kdtree = spatial.cKDTree(xyz, 50)
            dd, _ = kdtree.query(xyz, k=30)
            k_dis = np.mean(dd, axis=1).astype(np.float32)
        point_num = len(points)
        n = point_num // self.size
        distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        intensity = points[:, 3]
        labels = points[:, -1].astype(int)
        labels[labels != 2] = 1

        order = np.arange(0, n * self.size)
        if point_num % self.size > self.H:
            end = point_num - point_num % self.H
            start = (2 * n + 1) * self.size - end
            res_order = np.arange(end - self.size, end)
            order = np.append(order, res_order)
            order = order.reshape((n + 1, self.W, self.H)).transpose(0, 2, 1).astype(int)
        else:
            start = -1
            order = order.reshape((n, self.W, self.H)).transpose(0, 2, 1).astype(int)

        label_1 = labels[order]
        label = torch.as_tensor(label_1)
        distance = torch.as_tensor(distance[order])
        reflectivity = torch.as_tensor(intensity[order])
        if self.k_dis:
            k_dis = torch.as_tensor(k_dis[order])
            if self.transform:
                for i in range(len(distance)):
                    [distance[i], reflectivity[i], k_dis[i]], label[i] = self.transform(
                        [distance[i], reflectivity[i], k_dis[i]], label[i])
                    # [distance[i], reflectivity[i], k_dis[i]] = self.transform([distance[i], reflectivity[i], k_dis[i]])
            param_lists = [distance, reflectivity, k_dis]
        else:
            if self.transform:
                for i in range(len(distance)):
                    [distance[i], reflectivity[i]], label[i] = self.transform([distance[i], reflectivity[i]], label[i])
            param_lists = [distance, reflectivity]

        return param_lists, label, file_id, start
        # return distance, reflectivity, k_dis, label, file_id, start

    def __len__(self):
        return len(self.lidar[0])

    @staticmethod
    def num_classes():
        return len(DENSE.classes)

    @staticmethod
    def mean():
        return [24.03, 10.67]

    @staticmethod
    def std():
        return [16.55, 14.00]

    @staticmethod
    def class_weights():
        # return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])
        return torch.tensor([1 / 15.0, 10.0, 10.0])
