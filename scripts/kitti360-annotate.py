import os
from os import listdir
import pickle
import shutil
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

img_type_0 = cv2.imread('../wiki/images/0.png')
img_type_1 = cv2.imread('../wiki/images/1.png')
img_type_2 = cv2.imread('../wiki/images/2.png')
img_type_3 = cv2.imread('../wiki/images/3.png')
img_type_4 = cv2.imread('../wiki/images/4.png')
img_type_5 = cv2.imread('../wiki/images/5.png')
img_type_6 = cv2.imread('../wiki/images/6.png')

scale_percent = 200  # percent of original size
width = int(img_type_0.shape[1] * scale_percent / 100)
height = int(img_type_0.shape[0] * scale_percent / 100)
dim = (width, height)
img_type_0 = cv2.resize(img_type_0, dim, interpolation = cv2.INTER_AREA)
img_type_1 = cv2.resize(img_type_1, dim, interpolation = cv2.INTER_AREA)
img_type_2 = cv2.resize(img_type_2, dim, interpolation = cv2.INTER_AREA)
img_type_3 = cv2.resize(img_type_3, dim, interpolation = cv2.INTER_AREA)
img_type_4 = cv2.resize(img_type_4, dim, interpolation = cv2.INTER_AREA)
img_type_5 = cv2.resize(img_type_5, dim, interpolation = cv2.INTER_AREA)
img_type_6 = cv2.resize(img_type_6, dim, interpolation = cv2.INTER_AREA)


def save_csv(annotations, filename="kitti360-crossings.cvs"):
    filename = os.path.join(base_folder, filename)
    with open(filename, 'w') as csv:
        for seq_ann, i in enumerate(annotations):
            for seq_file, j in enumerate(i):
                if j > -1:
                    out = files[seq_ann][seq_file] + ';' + str(j) + '\n'
                    # print(files[seq_ann][seq_file], ";", j)
                    # print(out)
                    csv.write(out)
    print("Annotations saved to ", filename)


def save_frames(where):
    if where:
        for seq_ann, i in enumerate(annotations):
            for seq_file, j in enumerate(i):
                if j > -1:
                    src = files[seq_ann][seq_file]
                    dst_folder = os.path.join(where,str(j))
                    file_prefix = [s for s in src.split('/') if "2013" in s][0]
                    file_suffix = os.path.split(src)[1]
                    dst = os.path.join(dst_folder, file_prefix + '_' + file_suffix)
                    if src != dst:
                        if not os.path.exists(dst_folder):
                            os.makedirs(dst_folder)
                        print("Copying ", src, " ", dst)
                        shutil.copy2(src, dst, follow_symlinks=False)
                    else:
                        print("src and dst files are the same, skipping... provide a good path please!")
        print("All files copied")


def print_help():
    print("Right Arrow -  next frame")
    print("Left Arrow  -  previous frame")
    print("Up Arrow    -  +10 frames")
    print("Down Arrow  -  -10 frames")
    print("space       -  reset frame to unknown")
    print("s           -  statistics")
    print("h           -  print this help")
    print("q           -  skip sequence")
    print("F12         -  exit")
    print("0..6 numbers for 0..6 intersection type")


def summary(annotations):
    print("Computing annotations...")
    type_0 = sum(sum(i == 0 for i in j) for j in annotations)
    type_1 = sum(sum(i == 1 for i in j) for j in annotations)
    type_2 = sum(sum(i == 2 for i in j) for j in annotations)
    type_3 = sum(sum(i == 3 for i in j) for j in annotations)
    type_4 = sum(sum(i == 4 for i in j) for j in annotations)
    type_5 = sum(sum(i == 5 for i in j) for j in annotations)
    type_6 = sum(sum(i == 6 for i in j) for j in annotations)

    print("Type 0: ", type_0)
    print("Type 1: ", type_1)
    print("Type 2: ", type_2)
    print("Type 3: ", type_3)
    print("Type 4: ", type_4)
    print("Type 5: ", type_5)
    print("Type 6: ", type_6)
    print("Overall: ", type_0 + type_1 + type_2 + type_3 + type_4 + type_5 + type_6, "\n")

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

# save_frames('/media/ballardini/4tb/KITTI-360/moved')

#save_csv(annotations)
print_help()
print("\nStart\n")

for sequence_number, sequence in enumerate(files):

    k = 1
    file = 0

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    while k is not 0:
        print(str(folders[sequence_number]) + " -- " + str(file) + "/" + str(len(sequence)))
        img = cv2.imread(sequence[file])

        v_pos = int(img.shape[0] - img_type_0.shape[0] - 10)
        h_pos = int(img.shape[1] / 2 - img_type_0.shape[1] / 2)
        if annotations[sequence_number][file] == 0:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_0
        elif annotations[sequence_number][file] == 1:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_1
        elif annotations[sequence_number][file] == 2:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_2
        elif annotations[sequence_number][file] == 3:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_3
        elif annotations[sequence_number][file] == 4:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_4
        elif annotations[sequence_number][file] == 5:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_5
        elif annotations[sequence_number][file] == 6:
            img[v_pos:v_pos + img_type_0.shape[0], h_pos:h_pos + img_type_0.shape[1]] = img_type_6

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
        if 48 <= k <= 55 and file < len(sequence)-1:
            file = file + 1

        if k == 32:  # deselect the frame
            annotations[sequence_number][file] = -1
            file = file + 1

        if k == 104:  # print help
            print_help()

        if k == 115:  # show statistics
            summary(annotations)

        with open(annotations_file, 'wb') as f:
            pickle.dump(annotations, f)

        if k == right and file+1 < len(sequence)-1:
            file = file + 1
        if k == up and file+10 < len(sequence)-1:
            file = file + 10

        if k == left and file > 0:
            file = file - 1
        if k == down and file-10 > 0:
            file = file - 10

        if k == 113:
            break

        if k == 201:
            cv2.destroyAllWindows()
            exit(-1)

    cv2.destroyAllWindows()

summary(annotations)
save_csv(annotations)