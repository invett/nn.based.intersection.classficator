"""
    TO ENABLE THE CREATION OF THE DATASET, ENABLE THE FOLLOWING LINE WITH SIMULATE=FALSE
    save_frames('/media/ballardini/4tb/KITTI-360/moved', simulate=False, mono=False)
    NEW!!! Use SAVING_CALLS instead of editing the TRUE/FALSE if ... better, not best. a small step for a man...

    Note: saving things is an ALTERNATIVE to editing the labels. So if this is TRUE, the program will ONLY save & exit

    NOTE 2: after labelling, when you go through all the images, the CSV will be saved! No need to add this call, this
            is needed only to recreate the dataset (or recreate the CSV from the pickle..)

    NOTE 3:

    code used to split old pickles in smaller ones 'per-folder'

    for indice, valore in enumerate(lista):
    annotations_ = annotations[0][indice]
    annotations_file_ = '/home/ballardini/Desktop/KITTI-ROAD/' + valore + '.pickle'
    print(str(len(annotations_)) + annotations_file_)
    with open(annotations_file_, 'wb') as f:
        pickle.dump(annotations_, f)


"""
SAVING_CALLS = False

import tkinter as tk
from tkinter import simpledialog
import os
import pickle
import shutil
from os import listdir
import imutils
import json

import cv2
import numpy as np
import random
from miscellaneous.utils import split_dataset

# datasets: KITTI360 | ALCALA | OXFORD | KITTI-ROAD
# dataset = 'KITTI-ROAD'
dataset = 'KITTI360'
# dataset = 'ALCALA'
# dataset = 'AQP'
# dataset = 'alcala-26.01.2021'
# dataset = 'alcala-12.02.2021'

# definitions, will be specialized later .. but just to avoid warnings
resizeme = 0

pickle_filenames = []
extract_field_from_path = -1

if dataset == 'KITTI-ROAD':
    base_folder = '/home/ballardini/Desktop/KITTI-ROAD/'
    extract_field_from_path = 14

    folders = ['2011_09_26_drive_0019_sync', '2011_09_26_drive_0020_sync', '2011_09_26_drive_0022_sync',
               '2011_09_26_drive_0023_sync', '2011_09_26_drive_0035_sync', '2011_09_26_drive_0036_sync',
               '2011_09_26_drive_0039_sync', '2011_09_26_drive_0046_sync', '2011_09_26_drive_0061_sync',
               '2011_09_26_drive_0064_sync', '2011_09_26_drive_0079_sync', '2011_09_26_drive_0086_sync',
               '2011_09_26_drive_0087_sync', '2011_09_30_drive_0018_sync', '2011_09_30_drive_0020_sync',
               '2011_09_30_drive_0027_sync', '2011_09_30_drive_0028_sync', '2011_09_30_drive_0033_sync',
               '2011_09_30_drive_0034_sync', '2011_10_03_drive_0027_sync', '2011_10_03_drive_0034_sync']

    pickle_filenames = ['2011_09_26_drive_0019_sync.pickle', '2011_09_26_drive_0020_sync.pickle',
                        '2011_09_26_drive_0022_sync.pickle', '2011_09_26_drive_0023_sync.pickle',
                        '2011_09_26_drive_0035_sync.pickle', '2011_09_26_drive_0036_sync.pickle',
                        '2011_09_26_drive_0039_sync.pickle', '2011_09_26_drive_0046_sync.pickle',
                        '2011_09_26_drive_0061_sync.pickle', '2011_09_26_drive_0064_sync.pickle',
                        '2011_09_26_drive_0079_sync.pickle', '2011_09_26_drive_0086_sync.pickle',
                        '2011_09_26_drive_0087_sync.pickle', '2011_09_30_drive_0018_sync.pickle',
                        '2011_09_30_drive_0020_sync.pickle', '2011_09_30_drive_0027_sync.pickle',
                        '2011_09_30_drive_0028_sync.pickle', '2011_09_30_drive_0033_sync.pickle',
                        '2011_09_30_drive_0034_sync.pickle', '2011_10_03_drive_0027_sync.pickle',
                        '2011_10_03_drive_0034_sync.pickle']

    csv_filenames = ['2011_09_26_drive_0019_sync.csv', '2011_09_26_drive_0020_sync.csv',
                     '2011_09_26_drive_0022_sync.csv', '2011_09_26_drive_0023_sync.csv',
                     '2011_09_26_drive_0035_sync.csv', '2011_09_26_drive_0036_sync.csv',
                     '2011_09_26_drive_0039_sync.csv', '2011_09_26_drive_0046_sync.csv',
                     '2011_09_26_drive_0061_sync.csv', '2011_09_26_drive_0064_sync.csv',
                     '2011_09_26_drive_0079_sync.csv', '2011_09_26_drive_0086_sync.csv',
                     '2011_09_26_drive_0087_sync.csv', '2011_09_30_drive_0018_sync.csv',
                     '2011_09_30_drive_0020_sync.csv', '2011_09_30_drive_0027_sync.csv',
                     '2011_09_30_drive_0028_sync.csv', '2011_09_30_drive_0033_sync.csv',
                     '2011_09_30_drive_0034_sync.csv', '2011_10_03_drive_0027_sync.csv',
                     '2011_10_03_drive_0034_sync.csv']

    width = 1408
    height = 376
    position1 = (10, 30)
    position2 = (950, 30)
    position3 = (950, 60)
    resizeme = 0  # resizeme = 0 does not perform the resize

if dataset == 'KITTI360':
    base_folder = '/media/ballardini/4tb/ALVARO/KITTI-360/'
    extract_field_from_path = 19

    folders = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync',
               '2013_05_28_drive_0004_sync', '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync',
               '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync', '2013_05_28_drive_0010_sync']

    pickle_filenames = ['2013_05_28_drive_0000_sync.pickle', '2013_05_28_drive_0002_sync.pickle',
                        '2013_05_28_drive_0003_sync.pickle', '2013_05_28_drive_0004_sync.pickle',
                        '2013_05_28_drive_0005_sync.pickle', '2013_05_28_drive_0006_sync.pickle',
                        '2013_05_28_drive_0007_sync.pickle', '2013_05_28_drive_0009_sync.pickle',
                        '2013_05_28_drive_0010_sync.pickle']

    csv_filenames = ['2013_05_28_drive_0000_sync.csv', '2013_05_28_drive_0002_sync.csv',
                        '2013_05_28_drive_0003_sync.csv', '2013_05_28_drive_0004_sync.csv',
                        '2013_05_28_drive_0005_sync.csv', '2013_05_28_drive_0006_sync.csv',
                        '2013_05_28_drive_0007_sync.csv', '2013_05_28_drive_0009_sync.csv',
                        '2013_05_28_drive_0010_sync.csv']

    width = 1408
    height = 376
    position1 = (10, 30)
    position2 = (1000, 30)
    position3 = (1000, 60)

    resizeme = 0  # resizeme = 0 does not perform the resize

if dataset == 'ALCALA':
    # images from raw files are 1920x1200 - resize as needed.
    # ffmpeg -f rawvideo -pixel_format bayer_rggb8 -video_size 1920x^C00 -framerate 10 -i R2_video_0002_camera2.raw -vf
    # scale=800:-1 R2_video_0002_camera2_png/%010d.png
    base_folder = '/home/ballardini/Desktop/ALCALA/'

    # folders = ['R2_video_0002_camera2_png']
    # csv_filename = "R2C2.csv"
    # pickle_filename = 'R2C2.pickle'

    # folders = ['R2_video_0002_camera1_png']  # 36608 reverse gear
    # csv_filename = "R2C1.csv"
    # pickle_filename = 'R2C1.pickle'

    folders = ['R1_video_0002_camera1_png']  # 8944 imgs
    csv_filename = "R1C1.csv"
    pickle_filename = 'R1C1.pickle'

    height = 500
    position1 = (10, 30)
    position2 = (500, 30)
    position3 = (500, 60)
    resizeme = 0  # resizeme = 0 does not perform the resize
    width = 800

if dataset == 'alcala-26.01.2021':
    # images from raw files are 1920x1200 - resize as needed.
    # ffmpeg -f rawvideo -pixel_format bayer_rggb8 -video_size 1920x^C00 -framerate 10 -i R2_video_0002_camera2.raw -vf
    # scale=800:-1 R2_video_0002_camera2_png/%010d.png
    base_folder = '/home/ballardini/Desktop/alcala-26.01.2021/'
    extract_field_from_path = 8

    folders = ['161604AA', '161657AA', '161957AA', '162257AA', '162557AA', '162857AA', '163157AA', '163457AA',
               '163757AA', '164057AA', '164357AA', '164657AA', '164957AA', '165257AA', '165557AA', '165857AA',
               '170157AA', '170457AA', '170757AA', '171057AA', '171357AA', '171657AA', '171957AA', '172257AA',
               '172557AA', '172857AA', '173158AA', '173457AA', '173757AA', '174057AA', '174357AA', '174657AA',
               '174957AA', '175258AA', '175557AA', '175857AA', '180158AA', '180458AA', '180757AA', '181058AA',
               '181358AA', '181658AA', '181958AA', '182258AA', '182558AA', '182858AA', '183158AA', '183458AA',
               '183758AA', '184058AA', '184358AA', '184658AA']

    pickle_filenames = ['161604AA.pickle', '161657AA.pickle', '161957AA.pickle', '162257AA.pickle', '162557AA.pickle',
                        '162857AA.pickle', '163157AA.pickle', '163457AA.pickle', '163757AA.pickle', '164057AA.pickle',
                        '164357AA.pickle', '164657AA.pickle', '164957AA.pickle', '165257AA.pickle', '165557AA.pickle',
                        '165857AA.pickle', '170157AA.pickle', '170457AA.pickle', '170757AA.pickle', '171057AA.pickle',
                        '171357AA.pickle', '171657AA.pickle', '171957AA.pickle', '172257AA.pickle', '172557AA.pickle',
                        '172857AA.pickle', '173158AA.pickle', '173457AA.pickle', '173757AA.pickle', '174057AA.pickle',
                        '174357AA.pickle', '174657AA.pickle', '174957AA.pickle', '175258AA.pickle', '175557AA.pickle',
                        '175857AA.pickle', '180158AA.pickle', '180458AA.pickle', '180757AA.pickle', '181058AA.pickle',
                        '181358AA.pickle', '181658AA.pickle', '181958AA.pickle', '182258AA.pickle', '182558AA.pickle',
                        '182858AA.pickle', '183158AA.pickle', '183458AA.pickle', '183758AA.pickle', '184058AA.pickle',
                        '184358AA.pickle', '184658AA.pickle']

    csv_filenames = ['161604AA.csv', '161657AA.csv', '161957AA.csv', '162257AA.csv', '162557AA.csv', '162857AA.csv',
                     '163157AA.csv', '163457AA.csv', '163757AA.csv', '164057AA.csv', '164357AA.csv', '164657AA.csv',
                     '164957AA.csv', '165257AA.csv', '165557AA.csv', '165857AA.csv', '170157AA.csv', '170457AA.csv',
                     '170757AA.csv', '171057AA.csv', '171357AA.csv', '171657AA.csv', '171957AA.csv', '172257AA.csv',
                     '172557AA.csv', '172857AA.csv', '173158AA.csv', '173457AA.csv', '173757AA.csv', '174057AA.csv',
                     '174357AA.csv', '174657AA.csv', '174957AA.csv', '175258AA.csv', '175557AA.csv', '175857AA.csv',
                     '180158AA.csv', '180458AA.csv', '180757AA.csv', '181058AA.csv', '181358AA.csv', '181658AA.csv',
                     '181958AA.csv', '182258AA.csv', '182558AA.csv', '182858AA.csv', '183158AA.csv', '183458AA.csv',
                     '183758AA.csv', '184058AA.csv', '184358AA.csv', '184658AA.csv']

    height = 500
    position1 = (10, 30)
    position2 = (500, 30)
    position3 = (500, 60)
    resizeme = 0  # resizeme = 0 does not perform the resize
    width = 800

if dataset == 'alcala-12.02.2021':
    # images from raw files are 1920x1200 - resize as needed.
    # ffmpeg -f rawvideo -pixel_format bayer_rggb8 -video_size 1920x^C00 -framerate 10 -i R2_video_0002_camera2.raw -vf
    # scale=800:-1 R2_video_0002_camera2_png/%010d.png
    base_folder = '/home/ballardini/Desktop/alcala-12.02.2021.000/'
    extract_field_from_path = 10

    folders = ['120445AA', '122302AA', '164002AA', '165810AA']

    pickle_filenames = ['alcala-12.02.2021.120445AA.pickle',
                        'alcala-12.02.2021.122302AA.pickle',
                        'alcala-12.02.2021.164002AA.pickle',
                        'alcala-12.02.2021.165810AA.pickle']

    csv_filenames = ['alcala-12.02.2021.120445AA.csv',
                     'alcala-12.02.2021.122302AA.csv',
                     'alcala-12.02.2021.164002AA.csv',
                     'alcala-12.02.2021.165810AA.csv']

    height = 500
    position1 = (10, 30)
    position2 = (500, 30)
    position3 = (500, 60)
    resizeme = 0  # resizeme = 0 does not perform the resize
    width = 800

if dataset == 'OXFORD':
    # images from raw files are 1920x1200 - resize as needed.
    # ffmpeg -f rawvideo -pixel_format bayer_rggb8 -video_size 1920x^C00 -framerate 10 -i R2_video_0002_camera2.raw -vf
    # scale=800:-1 R2_video_0002_camera2_png/%010d.png
    base_folder = '/media/ballardini/7D3AD71E1EACC626/DATASET/JournalVersion/OXFORD/oxford_img_bb'
    folders = []
    [folders.append(x[0]) for x in os.walk(base_folder) if 'undistort' in x[0]]
    width = 320
    height = 240
    position1 = (10, 30)
    position2 = (100, 30)
    position3 = (100, 60)
    csv_filename = "oxford-crossings.csv"
    pickle_filename = 'oxford-annotations.pickle'
    resizeme = 800

if dataset == 'AQP':
    # images from raw files are 1920x1200 - resize as needed.
    # ffmpeg -f rawvideo -pixel_format bayer_rggb8 -video_size 1920x^C00 -framerate 10 -i R2_video_0002_camera2.raw -vf
    # scale=800:-1 R2_video_0002_camera2_png/%010d.png
    base_folder = '/home/ballardini/Desktop/AQP/'

    folders = ['2014_0101_004025_115', '2014_0101_034707_116', '2014_0101_221527_116', '2014_0102_102920_117',
               '2020_1101_144951_003', '2020_1101_153807_018', '2020_1104_090905_021', '2020_1104_091505_023',
               '2020_1104_094943_028', '2020_1104_133551_032', '2020_1104_133852_033', '2020_1104_135353_038',
               '2020_1104_141259_039', '2020_1104_165659_057', '2020_1104_172359_066']

    csv_filename = "aqp.csv"
    pickle_filename = 'aqp.pickle'

    height = 500
    position1 = (10, 30)
    position2 = (500, 30)
    position3 = (500, 60)
    resizeme = 0  # resizeme = 0 does not perform the resize
    width = 800

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

# folder_2013_05_28_drive_0000_sync = []
# folder_2013_05_28_drive_0002_sync = []
# folder_2013_05_28_drive_0003_sync = []
# folder_2013_05_28_drive_0004_sync = []
# folder_2013_05_28_drive_0005_sync = []
# folder_2013_05_28_drive_0006_sync = []
# folder_2013_05_28_drive_0007_sync = []
# folder_2013_05_28_drive_0009_sync = []
# folder_2013_05_28_drive_0010_sync = []

left = 81  # 2424832
up = 82
right = 83  # 2555904
down = 84
space = 32
f12 = 201

files = []

img_type_0 = cv2.imread('../wiki/images/0.png')
img_type_1 = cv2.imread('../wiki/images/1.png')
img_type_2 = cv2.imread('../wiki/images/2.png')
img_type_3 = cv2.imread('../wiki/images/3.png')
img_type_4 = cv2.imread('../wiki/images/4.png')
img_type_5 = cv2.imread('../wiki/images/5.png')
img_type_6 = cv2.imread('../wiki/images/6.png')
img_type_all = cv2.imread('../wiki/images/all.png')

scale_percent = 200  # percent of original size
width = int(img_type_0.shape[1] * scale_percent / 100)
height = int(img_type_0.shape[0] * scale_percent / 100)
dim = (width, height)
img_type_0 = cv2.resize(img_type_0, dim, interpolation=cv2.INTER_AREA)
img_type_1 = cv2.resize(img_type_1, dim, interpolation=cv2.INTER_AREA)
img_type_2 = cv2.resize(img_type_2, dim, interpolation=cv2.INTER_AREA)
img_type_3 = cv2.resize(img_type_3, dim, interpolation=cv2.INTER_AREA)
img_type_4 = cv2.resize(img_type_4, dim, interpolation=cv2.INTER_AREA)
img_type_5 = cv2.resize(img_type_5, dim, interpolation=cv2.INTER_AREA)
img_type_6 = cv2.resize(img_type_6, dim, interpolation=cv2.INTER_AREA)

scale_percent = 50  # percent of original size
width = int(img_type_all.shape[1] * scale_percent / 100)
height = int(img_type_all.shape[0] * scale_percent / 100)
dim = (width, height)
img_type_all = cv2.resize(img_type_all, dim, interpolation=cv2.INTER_AREA)


def hasNumbers(inputString):
    return all(char.isdigit() for char in inputString)


def save_csv(annotations, save_folder='/tmp'):
    """

    Args:
        annotations: input data
        filename: where to save the CSV

    Returns:

    the generated CSV has the structure of the original CSV from the previous IRALAB KITTI annotations, i.e.:

    framename ; distance-to-the-intersection ; label/class [0-6] ; in-crossing flag

    NOTICE that distance-to-the-intersections *DOES NOT* have any sense here, so we put -1 to avoid using a specific
    distance (that should be always positive).

    """
    for sequence_number_, i in enumerate(annotations):
        filename = os.path.join(save_folder, csv_filenames[sequence_number_])
        with open(filename, 'w') as csv:
            for sequence_file, j in enumerate(i):
                if j > -1:
                    # according with the documentation: '0000003374;-1;4;0'
                    out = os.path.splitext(os.path.split(files[sequence_number_][sequence_file])[1])[0] + ';-1;' + str(
                        j) + ';0\n'
                    csv.write(out)
        print("Annotations saved to ", csv_filenames[sequence_number_])


def save_frames(where, simulate=True, mono=True):
    """

    creates a copy of the selected frames. pass a destination folder, folder structure will be created.

    Args:
        where: destination folder
        simulate: if TRUE, just print what the routine would do. for debuggin' purposes.
        mono: where to create a MONOCULAR or STEREO dataset

    Returns: nothing, just do the work ...

    """
    _simulate = simulate
    _mono = mono
    cameras = []
    if _mono:
        cameras = ["image_00"]
        LR = ["left"]
    else:
        cameras = ["image_00", "image_01"]
        LR = ["left", "right"]

    images = 0
    for camera, leftright in zip(cameras, LR):
        if where:
            for seq_ann, i in enumerate(annotations):
                for seq_file, j in enumerate(i):
                    if j > -1:
                        src = files[seq_ann][seq_file]
                        src = src.replace("image_00",
                                          camera)  # replace image_00; does nothing otherwise; here the origin png
                        dst_folder = os.path.join(where, str(j), leftright)  # setup destination folder
                        if dataset == 'KITTI360':
                            # from
                            # '/media/ballardini/4tb/ALVARO/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000193.png'
                            # extrats:
                            # '2013_05_28_drive_0000_sync'
                            file_prefix = [s for s in src.split('/') if "2013" in s][0]
                        if dataset == 'alcala-26.01.2021':
                            file_prefix = src.split('/')[5]
                        file_suffix = os.path.split(src)[1]
                        dst = os.path.join(dst_folder, file_prefix + '_' + file_suffix)
                        if src != dst:
                            if not os.path.exists(dst_folder):
                                os.makedirs(dst_folder)
                            print("Copying ", src, " ", dst)
                            images = images + 1
                            if not _simulate:
                                shutil.copy2(src, dst, follow_symlinks=False)
                        else:
                            print("src and dst files are the same, skipping... provide a good path please!")

    print('All {} files copied'.format(images))


def print_help():
    """
    gives some help

    Returns: gives some help

    """
    print("F1                -  enable frame skipping")
    print("F2                -  disable frame skipping")
    print("F3 | Left Arrow   -  previous frame")
    print("F4 | Right Arrow  -  next frame")
    print("Up Arrow          -  +10 frames")
    print("Down Arrow        -  -10 frames")
    print("space             -  reset frame to unknown")
    print("s                 -  statistics")
    print("h                 -  print this help")
    print("q                 -  skip/next sequence | the CSV will be generated/updated")
    print("g                 -  goto specific frame")
    print("F12               -  exit | the CSV will NOT be generated/updated")
    print("0..6 numbers for 0..6 intersection type")

folders.sort()

# read the folders provided, please, only PNG files
for folder in folders:
    path = ''
    if dataset == 'KITTI-ROAD':
        path = os.path.join(base_folder, folder, 'image_02/data')
    if dataset == 'KITTI360':
        path = os.path.join(base_folder, 'data_2d_raw', folder, 'image_00/data_rect')
    if dataset == 'ALCALA':
        path = os.path.join(base_folder, folder)
    if dataset == 'alcala-26.01.2021':
        path = os.path.join(base_folder, folder)
    if dataset == 'alcala-12.02.2021':
        path = os.path.join(base_folder, 'RGB', folder)
    if dataset == 'AQP':
        path = os.path.join(base_folder, folder, 'image_02')
    if dataset == 'OXFORD':
        path = os.path.join(base_folder, folder)
    # list all files ending in .png
    files.append(sorted([path + '/' + f for f in listdir(path) if f.endswith('.png')]))

# if for some reason some of the folders is empty, say no.
for file_list_check in files:
    if len(file_list_check) == 1:
        print("The following folder has no images: ", os.path.split(file_list_check[0])[0])
        exit(-1)

annotations = []
annotations_filenames = []
pickle_filenames.sort()
if csv_filenames:
    csv_filenames.sort()
else:
    print("CSV filenames list was not provided - no csv will be saved!")

if pickle_filenames:
    for pickle_filename in pickle_filenames:
        # try to read all the pickles in the list
        annotations_file = os.path.join(base_folder, pickle_filename)
        if os.path.exists(annotations_file):
            with open(annotations_file, 'rb') as f:
                annotations.append(pickle.load(f))
                annotations_filenames.append(annotations_file)
        else:
            if not annotations:
                # create the pickle(s)
                for pickle_filename_ in pickle_filenames:
                    for sequence in files:
                        annotations.append(np.ones(len(sequence), dtype=np.int8) * -1)
                    with open(annotations_file, 'wb') as f:
                        pickle.dump(annotations, f)
                        annotations_filenames.append(os.path.join(base_folder, pickle_filename_))
            else:
                annotations_file = os.path.join(base_folder, pickle_filename)
                print('At least one of the provided pickles file is missing, so we won\'t continue')
                print(annotations_file)
                exit(2)
else:
    print("No pickle_filenames provided")
    exit(3)

annotations_file = ''  # should not be used anymore
f = ''

# ENABLE THIS LINE TO CREATE THE DATASET, IE, CREATE A NEW FOLDER STRUCTURE WITH DATA
if SAVING_CALLS:
    # save_frames('/home/ballardini/Desktop/alcala-26.01.2021-kitti360like', simulate=False, mono=True)
    # save_frames('/media/ballardini/4tb/KITTI-360/moved', simulate=False, mono=False)
    exit(1)

print_help()
print("\nStart\n")

skip = False

for sequence_number, sequence in enumerate(files):

    k = 1
    file = 0

    # the file sequence might start not from zero...
    start_number = int(os.path.splitext(os.path.basename(sequence[file]))[0])

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    while k != 0:
        print(str(folders[sequence_number]) + " -- " + str(file) + "/" + str(len(sequence)) + " -- " + str(
            sequence[file]))
        img = cv2.imread(sequence[file])

        # since datasets has different image shape, let put something easy to resize for labelling purposes
        if resizeme:
            img = imutils.resize(img, resizeme)

        v_pos = int(img.shape[0] - img_type_0.shape[0] - 10)
        h_pos = int(img.shape[1] / 2 - img_type_0.shape[1] / 2)
        v_pos_all = 80  # int(img_type_all.shape[0])
        h_pos_all = int(img.shape[1] / 2 - img_type_all.shape[1] / 2)

        img[v_pos_all:v_pos_all + img_type_all.shape[0], h_pos_all:h_pos_all + img_type_all.shape[1]] = img_type_all
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
        cv2.putText(img, str(file) + '/' + str(len(sequence) - 2), position2, font, fontScale, fontColor, lineType)
        cv2.putText(img, os.path.basename(sequence[file]), position3, font, fontScale, fontColor, lineType)

        cv2.imshow('image', img)

        if skip:
            if annotations[sequence_number][file] == -1:
                if file + 1 < len(sequence) - 1:
                    k = cv2.waitKey(1)
                    if k != 191:  # disable skip F2
                        file = file + 1
                        continue
        skip = False  # disable skipping once a valid frame is found

        k = cv2.waitKey(0)
        # print(k)

        if k == 190:  # enable skip F1
            if file + 1 < len(sequence) - 1:
                file = file + 1
            skip = True
        if k == 191:  # disable skip F2
            if file + 1 < len(sequence) - 1:
                file = file + 1
            skip = False

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
        if 48 <= k <= 55 and file < len(sequence) - 1:
            file = file + 1

        if k == 32:  # deselect the frame
            annotations[sequence_number][file] = -1
            file = file + 1

        if k == 104:  # print help
            print_help()

        if k == 115:  # show statistics
            split_dataset(annotations=annotations, files=files, extract_field_from_path=extract_field_from_path)

        with open(annotations_filenames[sequence_number], 'wb') as f:
            pickle.dump(annotations[sequence_number], f)

        # RIGHT or F4
        if (k == right or k == 193) and file + 1 < len(sequence) - 1:
            file = file + 1
        if k == up and file + 10 < len(sequence) - 1:
            file = file + 10

        # LEFT or F3
        if (k == left or k == 192) and file > 0:
            file = file - 1
            skip = False
        if k == down and file - 10 > 0:
            file = file - 10

        if k == ord('g'):
            ROOT = tk.Tk()
            ROOT.withdraw()
            while True:
                frame = simpledialog.askstring(title="KITTI360", prompt="Insert GOTO frame")
                if hasNumbers(frame):
                    frame = int(frame)
                else:
                    continue
                if 0 <= frame < len(sequence):
                    file = frame
                    break  # while True:  #     frame = int(input("Insert GOTO frame: "))  #     if 0 <= frame < len(sequence):  #         file = frame  #         break

        if k == 113:  # pressing q
            cv2.waitKey(10)
            cv2.destroyAllWindows()
            break

        if k == 201:  # pressing F12
            cv2.destroyAllWindows()
            split_dataset(annotations=annotations, files=files, extract_field_from_path=extract_field_from_path)
            save_csv(annotations)
            exit(-1)

    cv2.destroyAllWindows()

split_dataset(annotations=annotations, files=files, extract_field_from_path=extract_field_from_path)
save_csv(annotations)

