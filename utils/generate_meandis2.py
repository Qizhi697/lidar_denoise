import os
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm
import h5py

# dataset_dir = '/data/mayq/lidar/dataset/'
# train_velodyne_dir = os.path.join(dataset_dir, 'train', 'velodyne')
# val_velodyne_dir = os.path.join(dataset_dir, 'val', 'velodyne')
# train_ids = os.listdir(train_velodyne_dir)
# val_ids = os.listdir(val_velodyne_dir)
# train_files = [os.path.join(train_velodyne_dir, id) for id in train_ids]
# val_files = [os.path.join(val_velodyne_dir, id) for id in val_ids]
# total_files = train_files + val_files
# total_files.sort()
# for file in tqdm(total_files):
#     points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]
#     kdtree = spatial.cKDTree(points, 50)
#     dd, _ = kdtree.query(points, k=30)
#     mean_dis = np.mean(dd, axis=1)
#     filename = os.path.join(os.path.dirname(os.path.dirname(file)), 'meandis2', file.split('/')[-1].split('.')[0] + ".bin")
#     mean_dis.astype('float32').tofile(filename)

os.chdir('/data1/mayq/datasets/cnn_denoising/joke/')
files = [os.path.join(r, file) for r, d, f in os.walk('h5py') for file in f]
for file in tqdm(files):
    with h5py.File(file, "r", driver='core') as hdf5:
        sensorX = hdf5.get('sensorX_1')[()].reshape(-1)
        sensorY = hdf5.get('sensorY_1')[()].reshape(-1)
        sensorZ = hdf5.get('sensorZ_1')[()].reshape(-1)
        points = np.stack((sensorX, sensorY, sensorZ), axis=1)
        kdtree = spatial.cKDTree(points, 50)
        dd, _ = kdtree.query(points, k=30)
        mean_dis = np.mean(dd, axis=1)
        dir = os.path.join('knn', file.split('/')[1])
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = os.path.join(dir, file.split('/')[-1].split('.')[0] + '.bin')
        mean_dis.astype('float32').tofile(filename)
