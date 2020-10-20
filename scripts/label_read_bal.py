import csv
import os
import numpy as np
import shutil

root = '../../DualBiSeNet/data_raw/'
dst_folder = '/media/augusto/500GBHECTOR/augusto/kitti_2011'
distance = 20.0

folders = np.array([os.path.join(root, folder) for folder in sorted(os.listdir(root)) if
                    os.path.isdir(os.path.join(root, folder))])

missing = []

for folder in folders:
    file = os.path.join(folder, 'frames_topology.txt')
    with open(file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            # print(row[0])
            # continue
            if float(row[1]) < distance and row[3] != '1':
                base_folder = os.path.split(os.path.abspath(file))[0]
                src = os.path.join(os.path.split(os.path.abspath(file))[0], 'image_02', row[0] + '.png')

                dst_folder_ = os.path.join(dst_folder, str(row[2]))
                file_prefix = [s for s in src.split('/') if "2011" in s][0]
                file_suffix = os.path.split(src)[1]
                dst = os.path.join(dst_folder_, file_prefix + '_' + file_suffix)

                checkCreateDir = os.path.split(dst)[0]
                if not os.path.exists(checkCreateDir):
                    os.makedirs(checkCreateDir)

                if not os.path.isfile(src):
                    missing.append(src)
                    continue

                print("{:05.2f}m - Copying ".format(float(row[1])), src, " ", dst)
                shutil.copy2(src, dst, follow_symlinks=False)

print('##############################')

for i in missing:
    print('Take care! ', i, 'does not exist')

