"""
从train数据集中随机按比例生成train_joke，减少训练train的时间
"""
import os
import numpy as np
import shutil

dataset_root = 'dataset'
train_velodyne_dir = os.path.join(dataset_root, 'train', 'velodyne')
train_labels_dir = os.path.join(dataset_root, 'train', 'labels')
train_meandis2_dir = os.path.join(dataset_root, 'train', 'meandis2')
train_joke_velodyne_dir = os.path.join(dataset_root, 'train_joke', 'velodyne')
train_joke_labels_dir = os.path.join(dataset_root, 'train_joke', 'labels')
train_joke_meandis2_dir = os.path.join(dataset_root, 'train_joke', 'meandis2')
train_velodyne_bins = np.array(os.listdir(train_velodyne_dir))
train_labels_bins = np.array(os.listdir(train_labels_dir))
train_meandis2_bins = np.array(os.listdir(train_meandis2_dir))
train_velodyne_bins.sort()
train_labels_bins.sort()
train_meandis2_bins.sort()
num_ids = len(train_velodyne_bins)
train_joke_num = int(np.ceil(0.3 * num_ids))
train_joke_select = np.random.choice(num_ids, train_joke_num, replace=False)
train_joke_select.sort()
train_joke_velodyne_bins = train_velodyne_bins[train_joke_select]
train_joke_labels_bins = train_labels_bins[train_joke_select]
train_joke_meandis2_bins = train_meandis2_bins[train_joke_select]
for train_joke_velodyne_bin in train_joke_velodyne_bins:
    shutil.copy(os.path.join(train_velodyne_dir, train_joke_velodyne_bin), os.path.join(train_joke_velodyne_dir, train_joke_velodyne_bin))
for train_joke_labels_bin in train_joke_labels_bins:
    shutil.copy(os.path.join(train_labels_dir, train_joke_labels_bin), os.path.join(train_joke_labels_dir, train_joke_labels_bin))
for train_joke_meandis2_bin in train_joke_meandis2_bins:
    shutil.copy(os.path.join(train_meandis2_dir, train_joke_meandis2_bin), os.path.join(train_joke_meandis2_dir, train_joke_meandis2_bin))