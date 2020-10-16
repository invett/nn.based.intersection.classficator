import os
from os import listdir
import pickle

import cv2
import numpy as np

base_folder = '/media/ballardini/4tb/KITTI-360/'

folders = ['2013_05_28_drive_0000_sync',
           '2013_05_28_drive_0002_sync',
           '2013_05_28_drive_0003_sync',
           '2013_05_28_drive_0004_sync',
           '2013_05_28_drive_0005_sync',
           '2013_05_28_drive_0006_sync',
           '2013_05_28_drive_0007_sync',
           '2013_05_28_drive_0009_sync',
           '2013_05_28_drive_0010_sync']

folder_2013_05_28_drive_0000_sync = []
folder_2013_05_28_drive_0002_sync = []
folder_2013_05_28_drive_0003_sync = []
folder_2013_05_28_drive_0004_sync = []
folder_2013_05_28_drive_0005_sync = []
folder_2013_05_28_drive_0006_sync = []
folder_2013_05_28_drive_0007_sync = []
folder_2013_05_28_drive_0009_sync = []
folder_2013_05_28_drive_0010_sync = []

left = 81  # 2424832
up = 82
right = 83  # 2555904
down = 84
space = 32
f12 = 201

font = cv2.FONT_HERSHEY_SIMPLEX
position1 = (10, 30)
position2 = (1000, 30)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

files = []

for folder in folders:
    path = os.path.join(base_folder, 'data_2d_raw', folder, 'image_00/data_rect')
    # files.append(sorted([f for f in listdir(path) if isfile(join(path, f))]))
    files.append(sorted([path+'/'+f for f in listdir(path)]))

annotations = []
annotations_file = os.path.join(base_folder, 'annotations.pickle')

if os.path.exists(annotations_file):
    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f)
else:
    for sequence in files:
        annotations.append(np.ones(len(sequence), dtype=np.int8)*-1)
    with open(annotations_file, 'wb') as f:
        pickle.dump(annotations, f)

for sequence_number, sequence in enumerate(files):
    print(len(sequence))

    k = 1
    file = 0

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    while k is not 0:

        img = cv2.imread(sequence[file])

        cv2.putText(img, str(annotations[sequence_number][file]), position1, font, fontScale, fontColor, lineType)
        cv2.putText(img, str(file)+'/'+str(len(sequence)), position2, font, fontScale, fontColor, lineType)

        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        print(k)

        if k == 48:  # 1 as 0
            annotations[sequence_number][file] = 0
        if k == 49:  # 1 as 0
            annotations[sequence_number][file] = 1
        if k == 50:  # 1 as 1
            annotations[sequence_number][file] = 2
        if k == 51:  # 1 as 2
            annotations[sequence_number][file] = 3
        if k == 52:  # 1 as 3
            annotations[sequence_number][file] = 4
        if k == 53:  # 1 as 4
            annotations[sequence_number][file] = 5
        if k == 54:  # 1 as 5
            annotations[sequence_number][file] = 6
        if 48 <= k <= 55 and file < len(sequence):
            file = file + 1

        if k == 32:  # deselect the frame
            annotations[sequence_number][file] = -1
            file = file + 1


        with open(annotations_file, 'wb') as f:
            pickle.dump(annotations, f)

        if k == right and file+1 < len(sequence):
            file = file + 1
        if k == up and file+10 < len(sequence):
            file = file + 10

        if k == left and file-1 > 0:
            file = file - 1
        if k == down and file-10 > 0:
            file = file - 10

        if k == 113:
            break

        if k == 201:
            cv2.destroyAllWindows()
            exit(-1)

    cv2.destroyAllWindows()

