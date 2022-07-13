import os
import random
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import h5py

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
        # Class('snow2', 102, (119, 11, 32)),
    ]

    def __init__(self, root, split='train', transform=None, k_dis=False):
        # self.root = os.path.expanduser(root)
        self.root = root
        self.split = os.path.join(self.root, '{}'.format(split))
        self.transform = transform
        self.k_dis = k_dis
        self.lidar = []
        self.H = 64
        self.W = 400
        self.size = self.H * self.W
        if self.k_dis:
            self.data = {'distance': [], 'intensity': [], 'kdistance': []}
            for i, dir in enumerate(['velodyne', 'labels', 'meandis2']):
                self.lidar.append(
                    [os.path.join(r, file) for r, d, f in os.walk(os.path.join(self.split, dir)) for file in f])
            self.lidar[0].sort()
            self.lidar[1].sort()
            self.lidar[2].sort()
        else:
            self.data = {'distance': [], 'intensity': []}
            for i, dir in enumerate(['velodyne', 'labels']):
                self.lidar.append(
                    [os.path.join(r, file) for r, d, f in os.walk(os.path.join(self.split, dir)) for file in f])
            self.lidar[0].sort()
            self.lidar[1].sort()

    def __getitem__(self, index):
        points_file = self.lidar[0][index]
        file_id = points_file.split('/')[-1].split('.')[0]
        # labels_file = os.path.join(self.split, 'labels', file_id + '.label')
        points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(self.lidar[1][index], dtype=np.uint32).reshape(-1)
        if self.k_dis:
            k_dis = np.fromfile(self.lidar[2][index], dtype=np.float32).reshape(-1)
        point_num = len(labels)
        n = point_num // self.size
        distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        intensity = points[:, 3]
        # labels[(labels == 0)] = 0
        # labels[(labels == 110)] = 101
        # labels[(labels == 111)] = 102
        # labels[(labels != 0) & (labels != 101) & (labels != 102)] = 100

        # labels[(labels == 110) | (labels == 111)] = 101
        # labels[(labels != 101) & (labels != 0)] = 100

        labels[labels == 110] = 101
        labels[(labels != 101) & (labels != 0)] = 100
        index = np.arange(0, n * self.size)
        if point_num % self.size > self.H:
            end = point_num - point_num % self.H
            start = (2 * n + 1) * self.size - end
            res_index = np.arange(end - self.size, end)
            index = np.append(index, res_index)
            index = index.reshape((n + 1, self.W, self.H)).transpose(0, 2, 1).astype(int)
        else:
            start = -1
            index = index.reshape((n, self.W, self.H)).transpose(0, 2, 1).astype(int)

        label_dict = {0: 0, 100: 1, 101: 2}
        label_1 = np.vectorize(label_dict.get)(labels[index])
        label = torch.as_tensor(label_1)
        distance = torch.as_tensor(distance[index])
        reflectivity = torch.as_tensor(intensity[index])
        if self.k_dis:
            k_dis = torch.as_tensor(k_dis[index])
            if self.transform:
                for i in range(len(distance)):
                    [distance[i], reflectivity[i], k_dis[i]], label[i] = self.transform([distance[i], reflectivity[i], k_dis[i]], label[i])
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

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in DENSE.classes:
            cmap[cls.id, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=DENSE.mean(), std=DENSE.std())
    ])


    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    def visualize_seg(label_map, one_hot=False):
        if one_hot:
            label_map = np.argmax(label_map, axis=-1)

        out = np.zeros((label_map.shape[0], label_map.shape[1], 3))

        for l in range(1, DENSE.num_classes()):
            mask = label_map == l
            out[mask, 0] = np.array(DENSE.classes[l].color[1])
            out[mask, 1] = np.array(DENSE.classes[l].color[0])
            out[mask, 2] = np.array(DENSE.classes[l].color[2])

        return out


    dataset = DENSE('/data/mayq/WADS/dataset/', split='val', transform=joint_transforms)
    distance, reflectivity, label = random.choice(dataset)

    print('Distance size: ', distance.size())
    print('Reflectivity size: ', reflectivity.size())
    print('Label size: ', label.size())

    distance_map = Image.fromarray((255 * _normalize(distance.numpy())).astype(np.uint8))
    reflectivity_map = Image.fromarray((255 * _normalize(reflectivity.numpy())).astype(np.uint8))
    label_map = Image.fromarray((255 * visualize_seg(label.numpy())).astype(np.uint8))

    blend_map = Image.blend(distance_map.convert('RGBA'), label_map.convert('RGBA'), alpha=0.4)

    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.title("Distance")
    plt.imshow(distance_map)
    plt.subplot(222)
    plt.title("Reflectivity")
    plt.imshow(reflectivity_map)
    plt.subplot(223)
    plt.title("Label")
    plt.imshow(label_map)
    plt.subplot(224)
    plt.title("Result")
    plt.imshow(blend_map)

    plt.show()
