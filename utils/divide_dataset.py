"""
将WADS数据集七三分
"""
import os
import re
import numpy as np
import shutil

dataset_root = 'dataset'
train_velodyne_dir = os.path.join(dataset_root, 'train', 'velodyne')
train_labels_dir = os.path.join(dataset_root, 'train', 'labels')
val_velodyne_dir = os.path.join(dataset_root, 'val', 'velodyne')
val_labels_dir = os.path.join(dataset_root, 'val', 'labels')
sequences = list(filter(lambda x: re.match('^\d{2}', x) != None, os.listdir(dataset_root)))
for sequence in sequences:
    labels_dir = os.path.join(dataset_root, sequence, 'labels')
    velodyne_dir = os.path.join(dataset_root, sequence, 'velodyne')
    label_files = [os.path.join(dp, f) for dp, _, fn in os.walk(labels_dir) for f in fn]
    velodyne_files = [os.path.join(dp, f) for dp, _, fn in os.walk(velodyne_dir) for f in fn]
    label_files.sort()
    velodyne_files.sort()
    assert len(label_files) == len(velodyne_files)
    num_file = len(label_files)
    train_num = int(np.ceil(0.7 * num_file))
    train_ids = np.random.choice(num_file, train_num, replace=False)
    train_ids.sort()
    train_velodynes = [velodyne_files[i] for i in train_ids]
    train_labels = [label_files[i] for i in train_ids]
    val_velodynes = [velodyne_files[i] for i in range(0, num_file) if i not in train_ids]
    val_labels = [label_files[i] for i in range(0, num_file) if i not in train_ids]
    for train_velodyne in train_velodynes:
        shutil.copy(train_velodyne, os.path.join(train_velodyne_dir, train_velodyne.split('/')[-1]))
    for train_label in train_labels:
        shutil.copy(train_label, os.path.join(train_labels_dir, train_label.split('/')[-1]))
    for val_velodyne in val_velodynes:
        shutil.copy(val_velodyne, os.path.join(val_velodyne_dir, val_velodyne.split('/')[-1]))
    for val_label in val_labels:
        shutil.copy(val_label, os.path.join(val_labels_dir, val_label.split('/')[-1]))


