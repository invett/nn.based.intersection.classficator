import csv
import os
import numpy as np
import shutil

root = '/media/ballardini/4tb/KITTI-360/kitti360-augusto'
dst_folder = '/media/augusto/500GBHECTOR/augusto/kitti_2011'
distance = 20.0

folders = np.array([os.path.join(root, folder) for folder in sorted(os.listdir(root)) if
                    os.path.isdir(os.path.join(root, folder))])

list_2013_05_28_drive_0000_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0002_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0003_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0004_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0005_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0006_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0007_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0009_sync = [0, 0, 0, 0, 0, 0, 0]
list_2013_05_28_drive_0010_sync = [0, 0, 0, 0, 0, 0, 0]

overall = 0

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(root):
    if 'left' in root:
        for file in files:
            type = int(os.path.split(os.path.split(root)[0])[1])
            list = file[:26]
            if list == '2013_05_28_drive_0000_sync':
                list_2013_05_28_drive_0000_sync[type] += 1
            if list == '2013_05_28_drive_0002_sync':
                list_2013_05_28_drive_0002_sync[type] += 1
            if list == '2013_05_28_drive_0003_sync':
                list_2013_05_28_drive_0003_sync[type] += 1
            if list == '2013_05_28_drive_0004_sync':
                list_2013_05_28_drive_0004_sync[type] += 1
            if list == '2013_05_28_drive_0005_sync':
                list_2013_05_28_drive_0005_sync[type] += 1
            if list == '2013_05_28_drive_0006_sync':
                list_2013_05_28_drive_0006_sync[type] += 1
            if list == '2013_05_28_drive_0007_sync':
                list_2013_05_28_drive_0007_sync[type] += 1
            if list == '2013_05_28_drive_0009_sync':
                list_2013_05_28_drive_0009_sync[type] += 1
            if list == '2013_05_28_drive_0010_sync':
                list_2013_05_28_drive_0010_sync[type] += 1
            overall += 1

print('2013_05_28_drive_0000_sync: ', list_2013_05_28_drive_0000_sync)
print('2013_05_28_drive_0002_sync: ', list_2013_05_28_drive_0002_sync)
print('2013_05_28_drive_0003_sync: ', list_2013_05_28_drive_0003_sync)
print('2013_05_28_drive_0004_sync: ', list_2013_05_28_drive_0004_sync)
print('2013_05_28_drive_0005_sync: ', list_2013_05_28_drive_0005_sync)
print('2013_05_28_drive_0006_sync: ', list_2013_05_28_drive_0006_sync)
print('2013_05_28_drive_0007_sync: ', list_2013_05_28_drive_0007_sync)
print('2013_05_28_drive_0009_sync: ', list_2013_05_28_drive_0009_sync)
print('2013_05_28_drive_0010_sync: ', list_2013_05_28_drive_0010_sync)
print('overall: ', overall)