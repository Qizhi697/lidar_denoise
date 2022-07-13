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
        Class('clear', 100, (255, 0, 0)),
        Class('snow1', 101, (0, 0, 255)),
        # Class('snow2', 102, (119, 11, 32)),
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
        print("Loading Data")
        for i, label_file in enumerate(tqdm(self.lidar[1])):
            labels = np.fromfile(label_file, dtype=np.uint32).reshape(-1)
            points = np.fromfile(self.lidar[0][i], dtype=np.float32).reshape(-1, 4)
            if self.k_dis:
                k_dis = np.fromfile(self.lidar[2][i], dtype=np.float32).reshape(-1)
            points_num = len(labels)
            n = points_num // self.size
            distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
            intensity = points[:, 3]

            # labels[(labels == 0)] = 0
            # labels[(labels == 110)] = 101
            # labels[(labels == 111)] = 102
            # labels[(labels != 0) & (labels != 101) & (labels != 102)] = 100

            # labels[(labels == 110) | (labels == 111)] = 101
            # labels[(labels != 101) & (labels != 0)] = 100

            labels[labels == 110] = 101
            labels[(labels != 101)] = 100
            # 64c
            index = np.arange(0, n * self.size)
            if points_num % self.size > self.H:
                end = points_num - points_num % self.H
                res_index = np.arange(end - self.size, end)
                index = np.append(index, res_index)
                index = index.reshape((n + 1, self.W, self.H)).transpose(0, 2, 1).astype(int)
            else:
                index = index.reshape((n, self.W, self.H)).transpose(0, 2, 1).astype(int)

            labels = list(labels[index])
            distance = list(distance[index])
            intensity = list(intensity[index])
            self.label.extend(labels)
            self.data['distance'].extend(distance)
            self.data['intensity'].extend(intensity)
            if self.k_dis:
                k_distance = list(k_dis[index])
                self.data['kdistance'].extend(k_distance)

    def __getitem__(self, index):
        # label_1 = np.array(self.label[index])
        # distance_1 = np.array(self.data['distance'][index])
        # reflectivity_1 = np.array(self.data['intensity'][index])
        # k_distance_1 = np.array(self.data['kdistance'][index])
        # label_dict = {0: 0, 100: 1, 101: 2, 102: 3}
        label_dict = {0: 0, 100: 1, 101: 2}
        label_1 = np.vectorize(label_dict.get)(self.label[index])
        label = torch.as_tensor(label_1)
        distance = torch.as_tensor(self.data['distance'][index])
        reflectivity = torch.as_tensor(self.data['intensity'][index])
        if self.k_dis:
            k_distance = torch.as_tensor(self.data['kdistance'][index])
            param_lists = [distance, reflectivity, k_distance]
        else:
            param_lists = [distance, reflectivity]

        # distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
        # reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()
        # k_distance = torch.as_tensor(k_distance_1.astype(np.float32, copy=False)).contiguous()
        # label = torch.as_tensor(label_1.astype(np.float32, copy=False)).contiguous()
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

        for l in range(DENSE.num_classes()):
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
    label_map = Image.fromarray((visualize_seg(label.numpy())).astype(np.uint8))

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
