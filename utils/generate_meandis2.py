import os
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm

dataset_dir = '/data/mayq/lidar/dataset/'
train_velodyne_dir = os.path.join(dataset_dir, 'train', 'velodyne')
val_velodyne_dir = os.path.join(dataset_dir, 'val', 'velodyne')
train_ids = os.listdir(train_velodyne_dir)
val_ids = os.listdir(val_velodyne_dir)
train_files = [os.path.join(train_velodyne_dir, id) for id in train_ids]
val_files = [os.path.join(val_velodyne_dir, id) for id in val_ids]
total_files = train_files + val_files
total_files.sort()
for file in tqdm(total_files):
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]
    kdtree = spatial.cKDTree(points, 50)
    dd, _ = kdtree.query(points, k=30)
    mean_dis = np.mean(dd, axis=1)
    filename = os.path.join(os.path.dirname(os.path.dirname(file)), 'meandis2', file.split('/')[-1].split('.')[0] + ".bin")
    mean_dis.astype('float32').tofile(filename)
