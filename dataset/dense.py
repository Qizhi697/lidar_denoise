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
        Class('rain', 101, (220, 20, 60)),
        Class('fog', 102, (119, 11, 32)),
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
        self.data = {'distance': [], 'intensity': []}
        self.label = []
        self.lidar = [os.path.join(r, file) for r, d, f in os.walk(self.split) for file in f]
        print("Loading Data")
        label_dict = {0: 0, 100: 1, 101: 2, 102: 3}
        for i, file in enumerate(tqdm(self.lidar)):
            with h5py.File(file, "r", driver='core') as hdf5:
                distance_1 = hdf5.get('distance_m_1')[()]
                reflectivity_1 = hdf5.get('intensity_1')[()]
                label_1 = hdf5.get('labels_1')[()]
            label_1 = np.vectorize(label_dict.get)(label_1)
            self.label.append(label_1)
            self.data['distance'].append(distance_1)
            self.data['intensity'].append(reflectivity_1)

    def __getitem__(self, index):
        file_id = self.lidar[index].split('/')[-1].split('.')[0]
        label_1 = np.array(self.label[index])
        distance_1 = np.array(self.data['distance'][index])
        reflectivity_1 = np.array(self.data['intensity'][index])
        distance = torch.as_tensor(distance_1.astype(np.float32, copy=False)).contiguous()
        reflectivity = torch.as_tensor(reflectivity_1.astype(np.float32, copy=False)).contiguous()
        label = torch.as_tensor(label_1.astype(np.float32, copy=False)).contiguous()

        param_lists = [distance, reflectivity]
        if self.transform:
            param_lists, label = self.transform(param_lists, label)
        return param_lists, label, file_id, -1

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
        return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])

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
