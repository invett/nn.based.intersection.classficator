# WARNING: THIS SCRIPT WAS COPIED FROM THE <</PycharmProjects/reproject>> PROJECT !
# HAVE A LOOK TO <<generate.bev.and.pcds.sh>> BASH SCRIPT IN THE /scripts FOLDER (inside this project)

#
#
# python reproject.py 000000.png mask/000000.png out_aanet.ply out_aanet.pcd
#
#   1   aanet
#   2   alvaro
#   3   out ply (will be deleted)
#   4   out pcd
#

# the data structure for aanet is like this

#    augusto@ivan-pc:~/Datasets/KITTI/sequences/00/testing$ ll
#    total 420K
#    drwxrwxr-x 5 augusto augusto 4,0K jun 22 17:47 ./
#    drwxrwxr-x 3 augusto augusto 4,0K jun 19 13:19 ../
#    drwxrwxr-x 2 augusto augusto 132K jun 19 16:46 image_2/
#    drwxrwxr-x 2 augusto augusto 132K jun 19 16:29 image_3/
#    lrwxrwxrwx 1 augusto augusto    7 jun 22 17:47 left -> image_2/
#    drwxrwxr-x 2 augusto augusto 140K jun 22 18:29 pred/
#    lrwxrwxrwx 1 augusto augusto    7 jun 22 17:47 right -> image_3/
#


import os
import sys
import cv2
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from scipy.spatial.transform import Rotation as R
from mpl_toolkits import mplot3d
import time


visible = False
DOPCL = True  # create or not point clouds.

### data_folder_left = "data_road/training/image_2/"
### data_folder_right = "data_road/training_right/image_3/"
### data_folder_calib = "data_road/training/calib/"
### cat = ['uu', 'uum', 'um']
### IDX_LEN = 6
###
### idx_num = 1
### cat_idx = 2
### fname = cat[cat_idx]+'_'+str(idx_num).zfill(IDX_LEN)
### img_fname = fname + '.png'
### calib_fname = fname + '.txt'

# img_left_color = cv2.imread("image2_000000.png")
# img_right_color = cv2.imread("image3_000000.png")
img_left_color  = cv2.imread(sys.argv[5])
img_right_color = cv2.imread(sys.argv[6])
calib_fname = "calib.txt"

# img_left_bw = cv2.blur(cv2.cvtColor(img_left_color, cv2.COLOR_RGB2GRAY), (5, 5))
# img_right_bw = cv2.blur(cv2.cvtColor(img_right_color, cv2.COLOR_RGB2GRAY), (5, 5))

def showImg(img):
    if visible:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.show()


def write_ply(fn, verts, colors=0):
    if colors.any():
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    else:
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            end_header
            '''
    verts = verts.reshape(-1, 3)
    if colors.any():
        out_colors = colors.copy()
        verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        if colors.any():
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        else:
            np.savetxt(f, verts, fmt='%f %f %f ')


if visible:
    plt.imshow(img_right_bw, cmap='gray')
    plt.show()

####   #stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
####   stereo = cv2.StereoBM_create(numDisparities=192, blockSize=11)
####   disparity = stereo.compute(img_left_bw,img_right_bw)
####   cv2.imwrite("StereoBM.png", disparity)
####   img = disparity.copy()
####   plt.imshow(img, 'CMRmap_r')
####   if visible:
####       plt.show()

########################################################################################################################

# Reading calibration
matrix_type_1 = 'P2'
matrix_type_2 = 'P3'

calib_file = calib_fname
with open(calib_file, 'r') as f:
    fin = f.readlines()
    for line in fin:
        if line[:2] == matrix_type_1:
            calib_matrix_1 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)
        elif line[:2] == matrix_type_2:
            calib_matrix_2 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)


########################################################################################################################


# Calculate depth-to-disparity
cam1 = calib_matrix_1[:, :3]  # left image - P2
cam2 = calib_matrix_2[:, :3]  # right image - P3

Tmat = np.array([0.54, 0., 0.])
rev_proj_matrix = np.zeros((4, 4))

# TO GET THE Q-MATRIX, SAVED IN <<rev_proj_matrix>>
# cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
#                   distCoeffs1=0, distCoeffs2=0,
#                   imageSize=img_left_color.shape[:2],
#                   R=np.identity(3), T=Tmat,
#                   R1=None, R2=None,
#                   P1=None, P2=None, Q=rev_proj_matrix)

########################################################################################################################

# THIS IS THE Q MATRIX EVALUATED BEFORE and saved inside <<rev_proj_matrix>>
rev_proj_matrix=np.array([[   1.       ,     0.     ,       0.         , -607.19281006],
                          [   0.       ,     1.     ,       0.         , -185.21570587],
                          [   0.       ,     0.     ,       0.         ,  718.85601807],
                          [   0.       ,     0.     ,      -1.85185185 ,    0.        ]],dtype=np.float64)

#aanet=cv2.imread("000000.png",cv2.IMREAD_GRAYSCALE)
#aanet=cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)     #old version , loading png... now loading numpy directly instead
dict_data=load(sys.argv[1])
aanet=dict_data['arr_0']

start = time.time()

if visible:
    plt.imshow(aanet, 'CMRmap_r')
    plt.show()
#points = cv2.reprojectImageTo3D(img, rev_proj_matrix)
#points = cv2.reprojectImageTo3D(aanet.astype(np.float32)[:,:,0], rev_proj_matrix)
points = cv2.reprojectImageTo3D(aanet, rev_proj_matrix) #pass the Q matrix found above

#reflect on x axis
reflect_matrix = np.identity(3)
reflect_matrix[0] *= -1
points = np.matmul(points,reflect_matrix)

#extract colors from image
colors = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB)

#filter by min disparity
#mask = img > img.min()
out_points = points#[mask]
out_colors = colors#[mask]

# ALVARO MASK
#alvaro=cv2.imread("mask/000000.png",cv2.IMREAD_GRAYSCALE)
alvaro=cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
out_points = out_points[alvaro>0]
out_colors = out_colors[alvaro>0]

#filter by dimension
idx = np.fabs(out_points[:,2]) < 50.
out_points = out_points[idx]
out_colors = out_colors.reshape(-1, 3)
out_colors = out_colors[idx]

if DOPCL: # == > do-pcl, create the pcl files
    #name='out_aanet.ply'
    name=sys.argv[3]
    write_ply(name, out_points, out_colors) #WITH COLORS !!!!
    #write_ply(name, out_points) # NO COLOR!!! IF YOU WANT COLORS USE THE PREVIOUS LINE!!!
    print('%s saved' % name)

    #os.system("pcl_ply2pcd out_aanet.ply out_aanet.pcd")
    os.system("pcl_ply2pcd " + sys.argv[3] + " " + sys.argv[4])
    os.system("rm " + sys.argv[3])


### CREATE IMAGE

fx = 200#9.799200e+02
fy = 200#9.741183e+02
cx = 200#6.900000e+02
cy = 200#2.486443e+02
R_00 = np.array([ [1.000000e+00, 0.000000e+00, 0.000000e+00],
                  [0.000000e+00, 1.000000e+00, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=np.float64)
T_00 = np.array(  [0.000000e+00, 17.00000e+00, 10.500000e+00],  dtype=np.float64)
D_00 = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
K_00 = np.array([ [fx          , 0.000000e+00, cx],
                  [0.000000e+00, fy          , cy],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]] , dtype=np.float64)

rotatematrix1=R.from_euler('x',90,degrees=True).as_matrix()
rotatematrix2=R.from_euler('z',180,degrees=True).as_matrix()
rotatematrix3=R.from_euler('y',0,degrees=True).as_matrix()

reflect_matrix = np.identity(3)
reflect_matrix[0] *= -1
#out_points = np.matmul(out_points,reflect_matrix)
reflect_matrix = np.identity(3)
reflect_matrix[1] *= -1
out_points = np.matmul(out_points,reflect_matrix)
reflect_matrix = np.identity(3)
reflect_matrix[2] *= -1
out_points = np.matmul(out_points,reflect_matrix)

# Decimate
pointsandcolors = np.concatenate([out_points, out_colors],axis=1)
remaining_points = pointsandcolors.shape[0] * 1.0
pointsandcolors = pointsandcolors[np.random.choice(pointsandcolors.shape[0], int(remaining_points), replace=False), :]
out_points = pointsandcolors[:,:3].astype('float64')
out_colors = pointsandcolors[:,3:].astype('uint8')

imagePoints, jacobians = cv2.projectPoints(objectPoints=out_points, rvec=cv2.Rodrigues(R_00@rotatematrix1@rotatematrix3)[0], tvec=T_00, cameraMatrix=K_00, distCoeffs=D_00)

blank_image = np.zeros((int(cy*2),int(cx*2),3), np.uint8)
for pixel, color in zip(imagePoints, out_colors):
    if ((int(pixel[0, 1]) < blank_image.shape[0]) and (int(pixel[0, 0]) < blank_image.shape[1]) and
        (int(pixel[0, 1]) > 0) and (int(pixel[0, 0]) > 0)):
        blank_image[int(pixel[0, 1]), int(pixel[0, 0])] = color

end = time.time()
print(end - start)

#cv2.imshow('camera',imagePoints.squeeze())
cv2.imwrite(sys.argv[7], blank_image)
#cv2.waitKey(0)
