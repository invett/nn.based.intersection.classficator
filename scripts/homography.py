# importing code from HOMOGRAPHY lab practice by ballardini @ iralab

import os
from math import cos, pi, sin
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def getCameraRototraslation(pitchCorrection_, yawCorrection_, rollCorrection_, dx_, dy_, dz_):
    """
    Creates the extrinsic matrix using standard KITTI reference frame (z-forward ; x-right)

    Args:
        pitchCorrection_: correction on pitch/Y
        yawCorrection_: correction on yaw/Z
        rollCorrection_: correction on roll/X
        dx_: where the camera is x/y/z wrt ground
        dy_: where the camera is x/y/z wrt ground
        dz_: where the camera is x/y/z wrt ground

    Returns: the extrinsic camera matrix

    """
    rot = - pi / 2.0 + rollCorrection_
    R1 = np.array([[1, 0, 0, 0], [0, cos(rot), -sin(rot), 0], [0, sin(rot), cos(rot), 0], [0, 0, 0, 1]],
                  dtype=np.float32)
    rot = pi / 2.0 + yawCorrection_
    R2 = np.array([[cos(rot), 0, sin(rot), 0], [0, 1, 0, 0], [-sin(rot), 0, cos(rot), 0], [0, 0, 0, 1]],
                  dtype=np.float32)
    R3 = np.array([[1, 0, 0, 0], [0, cos(-pitchCorrection_), -sin(-pitchCorrection_), 0],
                   [0, sin(-pitchCorrection_), cos(-pitchCorrection_), 0], [0, 0, 0, 1]], dtype=np.float32)
    R = R1 @ R2 @ R3
    T = np.array([[1, 0, 0, dx_], [0, 1, 0, dy_], [0, 0, 1, dz_], [0, 0, 0, 1]], dtype=np.float32)
    RT = T @ R
    return RT


path = '/tmp'
filename = '0000000000.png'
image_path = os.path.join(path, filename)

pil_image = Image.open(image_path)

img = kornia.image_to_tensor(np.array(pil_image), keepdim=False)
points_src = torch.FloatTensor([[[527 - 100, 181], [598 + 100, 181], [1251 + 100, 229], [-340 - 100, 229], ]])
points_dst = torch.FloatTensor([[[0, 0], [224, 0], [224, 224], [0, 224], ]])

M = kornia.get_perspective_transform(points_src, points_dst)
data_warp = kornia.warp_perspective(img.float(), M, dsize=(224 + 30, 224 + 20))
warped = kornia.tensor_to_image(data_warp.byte())
plt.imshow(warped)
plt.show()

############################

K = np.array(
    [[9.786977e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00], [0.000000e+00, 9.717435e+02, 2.497222e+02, 0.000000e+00],
     [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]], dtype=np.float64)
dx = 6
dy = 0
dz = 2.0
points_3d = np.array([[16, 16, 120, 120], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
points_dst = torch.FloatTensor([[[0, 224], [224, 224], [224, 0], [0, 0], ]])
WorldToCam = np.linalg.inv(getCameraRototraslation(0.084, 0.1, 0, dx, dy, dz))
points_2d = K @ WorldToCam @ points_3d
points_2d = points_2d[:, :] / points_2d[2, :]
points_2d = points_2d[:2, :]
M = kornia.get_perspective_transform(
    torch.tensor(np.expand_dims(np.transpose(np.asarray(points_2d, dtype=np.float32)), axis=0)), points_dst)
data_warp = kornia.warp_perspective(img.float(), M, dsize=(224, 224))
warped = kornia.tensor_to_image(data_warp.byte())
plt.imshow(warped)
plt.show()

print("fine")  # plt.figure()
