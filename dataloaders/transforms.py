import torch
import numpy as np
import cv2.cv2 as cv2
from scipy.spatial.transform import Rotation as R
from skimage.transform import resize
# import matplotlib.pyplot as plt

# For debugging
ShowImage = False


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = resize(image, (new_h, new_w), anti_aliasing=True)

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class Normalize(object):

    def __call__(self, image):

        image = image / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean) / std

        return image.astype(np.float32)


class GenerateBev(object):
    """
    Data Augmentation routine;

    REFERENCE FRAME for rotation and translation:

        ROTATION PART:
        - z forward
        - x right
        - y down

            z (front of the camera)
           /
          /
         /
        |\
        | \
        |  \
        |   x (rightwards)
        |
        y (downwards)

        TRANSLATION PART:
            x: shift right (positive) or left (negative) the position of the vehicle with respect to the road centerline
            y: the higher the value, the vehicle will be more "inside" the crossing;
            z: the higher the value, the farther the camera will be from the ground leading to small crossing on image
    """

    def __init__(self,
                 maxdistance=50.0,
                 decimate=1.0,
                 random_Rx_value=2.0,
                 random_Ry_value=15.0,
                 random_Rz_value=2.0,
                 random_Tx=2.0,
                 random_Ty=2.0,
                 random_Tz=2.0
                 ):
        self.maxdistance = maxdistance
        self.decimate = decimate
        self.random_Rx_value = random_Rx_value
        self.random_Ry_value = random_Ry_value
        self.random_Rz_value = random_Rz_value
        self.random_Tx_value = random_Tx
        self.random_Ty_value = random_Ty
        self.random_Tz_value = random_Tz

    def __call__(self, sample):

        # this Q matrix was obtained using STEREORECTIFY; would be nice to import this part of the code too.
        rev_proj_matrix = np.array([
            [1., 0., 0., -607.19281006],
            [0., 1., 0., -185.21570587],
            [0., 0., 0., 718.85601807],
            [0., 0., -1.85185185, 0.]], dtype=np.float64)

        points = cv2.reprojectImageTo3D(sample['aanet'], rev_proj_matrix)

        # reflect on x axis
        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        points = np.matmul(points, reflect_matrix)

        colors = cv2.cvtColor(sample['image_02'], cv2.COLOR_BGR2RGB)

        # filter by min disparity
        # mask = img > img.min()
        out_points = points  # [mask]
        out_colors = colors  # [mask]

        # ALVARO MASK
        alvaro = sample['alvaromask']
        out_points = out_points[alvaro > 0]
        out_colors = out_colors[alvaro > 0]

        # filter by dimension
        idx = np.fabs(out_points[:, 2]) < self.maxdistance
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]

        # Create Virtual Camera for BEV generation
        fx = 200  # 9.799200e+02
        fy = 200  # 9.741183e+02
        cx = 200  # 6.900000e+02
        cy = 200  # 2.486443e+02

        # Set here a base rotation matrix for the camera orientation; will be rotated after
        R_00 = np.array([[1.000000e+00, 0.000000e+00, 0.000000e+00],
                         [0.000000e+00, 1.000000e+00, 0.000000e+00],
                         [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=np.float64)

        # Set here the default camera position of the camera
        random_Tx = np.random.uniform(-self.random_Tx_value,
                                      self.random_Tx_value)  # HERE Z is more to the RIGHT (pos val) or LEFT (neg val) wrt forward dir.
        random_Ty = np.random.uniform(-self.random_Ty_value,
                                      self.random_Ty_value)  # HERE Y is FORWARD/BACKWARD (closer or farther from the crossing)
        random_Tz = np.random.uniform(-self.random_Tz_value,
                                      self.random_Tz_value)  # HERE Z is the CAMERA HEIGHT (closer or farther from the ground)

        T_00 = np.array([0.000000e+00 + random_Tx,
                         17.00000e+00 + random_Ty,
                         10.50000e+00 + random_Tz], dtype=np.float64)

        # No distortion
        D_00 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Compose the intrinsic camera matrix using the parameters defined above
        K_00 = np.array([[fx, 0.000000e+00, cx],
                         [0.000000e+00, fy, cy],
                         [0.000000e+00, 0.000000e+00, 1.000000e+00]],
                        dtype=np.float64)

        baseRotationMatrix = R.from_euler('x', 90, degrees=True).as_matrix()

        # Noise for data augmentation, in DEGREES.
        # y-corresponds to a "yaw" of the image, the main effect we want (15 should be enough)
        random_Rx = np.random.uniform(-self.random_Rx_value, self.random_Rx_value)
        random_Ry = np.random.uniform(-self.random_Ry_value, self.random_Ry_value)
        random_Rz = np.random.uniform(-self.random_Rz_value, self.random_Rz_value)

        dataAugmentationRotationMatrixX = R.from_euler('x', random_Rx, degrees=True).as_matrix()
        dataAugmentationRotationMatrixY = R.from_euler('y', random_Ry, degrees=True).as_matrix()
        dataAugmentationRotationMatrixZ = R.from_euler('z', random_Rz, degrees=True).as_matrix()

        # Rotate the points by reflecting them on specific axes
        reflect_matrix = np.identity(3)
        reflect_matrix[1] *= -1
        out_points = np.matmul(out_points, reflect_matrix)

        reflect_matrix = np.identity(3)
        reflect_matrix[2] *= -1
        out_points = np.matmul(out_points, reflect_matrix)

        # Decimate the number of remaining points using the decimate parameter.
        pointsandcolors = np.concatenate([out_points, out_colors], axis=1)
        remaining_points = int(pointsandcolors.shape[0] * self.decimate)
        pointsandcolors = pointsandcolors[np.random.choice(pointsandcolors.shape[0], remaining_points, replace=False), :]
        out_points = pointsandcolors[:, :3].astype('float64')
        out_colors = pointsandcolors[:, 3:].astype('uint8')

        imagePoints, jacobians = cv2.projectPoints(objectPoints=out_points,
                                                   rvec=cv2.Rodrigues(R_00 @ baseRotationMatrix @
                                                                      dataAugmentationRotationMatrixX @
                                                                      dataAugmentationRotationMatrixY @
                                                                      dataAugmentationRotationMatrixZ)[0],
                                                   tvec=T_00, cameraMatrix=K_00, distCoeffs=D_00)

        # generate the image
        blank_image = np.zeros((int(cy * 2), int(cx * 2), 3), np.uint8)
        for pixel, color in zip(imagePoints, out_colors):
            if ((int(pixel[0, 1]) < blank_image.shape[0]) and
                    (int(pixel[0, 0]) < blank_image.shape[1]) and
                    (int(pixel[0, 1]) > 0) and
                    (int(pixel[0, 0]) > 0)):
                blank_image[int(pixel[0, 1]), int(pixel[0, 0])] = color

        if ShowImage:
            plt.imshow(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR))
            plt.show()

        return blank_image
