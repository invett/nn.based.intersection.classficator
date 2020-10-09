import torch
import numpy as np
import cv2.cv2 as cv2
from scipy.spatial.transform import Rotation as R
from skimage.transform import resize

# For debugging
ShowImage = False

if ShowImage:
    import matplotlib.pyplot as plt


class GenerateNewDataset(object):
    """
    This simply sets a value that will be used in the getitem to save the image (with the correct filename) inside the
    path passed as parameter
    """

    def __init__(self, path):
        self.path = path

    def __call__(self, sample):
        sample['path'] = self.path
        return sample


class WriteDebugInfoOnNewDataset(object):
    """
    Sets a label used later to write debug information
    """

    def __call__(self, sample):
        sample['debug'] = True
        return sample


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

    def __call__(self, sample):
        image = sample['data']
        if sample['generated_osm'] is not None:
            osm = sample['generated_osm']
        if sample['negative_osm'] is not None:
            osm_neg = sample['negative_osm']

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
        if sample['generated_osm'] is not None:
            osm = resize(osm, (new_h, new_w), anti_aliasing=True)
        if sample['negative_osm'] is not None:
            osm_neg = resize(osm_neg, (new_h, new_w), anti_aliasing=True)

        sample['data'] = image
        if sample['generated_osm'] is not None:
            sample['generated_osm'] = osm
        if sample['negative_osm'] is not None:
            sample['negative_osm'] = osm_neg

        return sample


class GrayScale(object):
    """Convert BGR images in GrayScale images"""

    def __call__(self, sample):
        image = sample['data']
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sample['data'] = image_gray

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['data'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # generated_osm might be optional, let's handle this
        if sample['generated_osm'] is not None:
            generated_osm = sample['generated_osm']
            generated_osm = generated_osm.transpose((2, 0, 1))
            if sample['negative_osm'] is not None:
                negative_osm = sample['negative_osm']
                negative_osm = negative_osm.transpose((2, 0, 1))
                return {'data': torch.from_numpy(image),
                        'generated_osm': torch.from_numpy(generated_osm).float(),
                        'negative_osm': torch.from_numpy(negative_osm).float(),
                        'label': label,
                        'image_path': sample['image_path']}
            else:
                return {'data': torch.from_numpy(image),
                        'generated_osm': torch.from_numpy(generated_osm).float(),
                        'label': label,
                        'image_path': sample['image_path']}
        else:
            return {'data': torch.from_numpy(image),
                    'label': label}


class Normalize(object):
    """
    Converts the range from 0..255 >> 0..1 (just to be used inside Pytorch)
    OPTIMIZE this might be faster directly in GPU/Tensor...
    """

    def __call__(self, sample):
        mean = np.array([0.062, 0.063, 0.064], dtype=np.float32)
        std = np.array([0.157, 0.156, 0.157], dtype=np.float32)
        image_norm = cv2.normalize(sample['data'], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
        image_norm = (image_norm - mean) / std
        sample['data'] = image_norm

        return sample


class NormalizePretrained(object):

    def __call__(self, sample):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = cv2.normalize(sample['image_02'], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
        image_norm = (image_norm - mean) / std

        sample['image_02'] = image_norm

        return sample


class Mirror(object):

    def __call__(self, sample):
        image, label = sample['data'], sample['label']

        if np.random.rand() > 0.5:

            flipped = cv2.flip(image, 1)
            if label == 1:
                label = 2
            elif label == 2:
                label = 1
            elif label == 3:
                label = 4
            elif label == 4:
                label = 3
            sample['data'] = flipped
            sample['label'] = label
            sample['mirrored'] = True
            return sample
        else:
            sample['mirrored'] = False
            return sample


class GenerateBev(object):
    """
    Data Augmentation routine;

    INPUT:
        1. AANET NPZ
        2. RGB IMAGE FROM KITTI
        3. ALVAROMASK

    OUTPUT:
        4. THE GENERATED BEV

    PARAMETERS:
        1. decimate factor - default=1.0 (no.decimate)
        2. TRANSLATION - random_T*_meters -> used to initialize a uniform distribution, then adding a value drawn from
        3. ROTATION - random_R*_degrees -> used to initialize a uniform distribution, then adding a value drawn from
        4. returnPoints --> used to return the 3D points with colors, used to debug the code

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
                 random_Rx_degrees=2.0,
                 random_Ry_degrees=15.0,
                 random_Rz_degrees=2.0,
                 random_Tx_meters=2.0,
                 random_Ty_meters=2.0,
                 random_Tz_meters=2.0,
                 returnPoints=False
                 ):
        self.maxdistance = maxdistance
        self.decimate = decimate
        self.random_Rx_degrees = random_Rx_degrees
        self.random_Ry_degrees = random_Ry_degrees
        self.random_Rz_degrees = random_Rz_degrees
        self.random_Tx_meters = random_Tx_meters
        self.random_Ty_meters = random_Ty_meters
        self.random_Tz_meters = random_Tz_meters
        self.returnPoints = returnPoints

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

        if self.returnPoints:
            save_out_points = out_points.copy()
            save_out_colors = out_colors.copy()

            # filter by dimension
            idx = np.fabs(save_out_points[:, :, 2]) < self.maxdistance
            save_out_points = save_out_points[idx]
            save_out_colors = save_out_colors
            save_out_colors = save_out_colors[idx]

        # ALVARO MASK # TODO this is the right place to disable ALVARO MASK s and so get the FULL - BEVs
        alvaro = sample['alvaromask']
        alvaro = np.ones(alvaro.shape, dtype=alvaro.dtype)  # furbata
        out_points = out_points[alvaro > 0]
        out_colors = out_colors[alvaro > 0]

        # filter by dimension
        #idx = np.fabs(out_points[:, 2]) < self.maxdistance
        #out_points = out_points[idx]
        #out_colors = out_colors.reshape(-1, 3)
        #out_colors = out_colors[idx]

        idx = np.fabs(out_points[:, 2]) < self.maxdistance
        out_points = out_points[idx]
        out_colors = out_colors[idx]
        idx = np.fabs(out_points[:, 1]) < 3.
        out_points = out_points[idx]
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
        # HERE Z is more to the RIGHT (pos val) or LEFT (neg val) wrt forward dir.
        # HERE Y is FORWARD/BACKWARD (closer or farther from the crossing)
        # HERE Z is the CAMERA HEIGHT (closer or farther from the ground)
        random_Tx = np.random.uniform(-self.random_Tx_meters, self.random_Tx_meters)
        random_Ty = np.random.uniform(-self.random_Ty_meters, self.random_Ty_meters)
        random_Tz = np.random.uniform(-self.random_Tz_meters, self.random_Tz_meters)

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
        random_Rx = np.random.uniform(-self.random_Rx_degrees, self.random_Rx_degrees)
        random_Ry = np.random.uniform(-self.random_Ry_degrees, self.random_Ry_degrees)
        random_Rz = np.random.uniform(-self.random_Rz_degrees, self.random_Rz_degrees)

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
        starting_points = pointsandcolors.shape[0]  # used for debug/reporting
        remaining_points = int(pointsandcolors.shape[0] * self.decimate)
        pointsandcolors = pointsandcolors[np.random.choice(pointsandcolors.shape[0], remaining_points, replace=False),
                          :]
        out_points = pointsandcolors[:, :3].astype('float64')
        out_colors = pointsandcolors[:, 3:].astype('float32')

        imagePoints, jacobians = cv2.projectPoints(objectPoints=out_points,
                                                   rvec=cv2.Rodrigues(R_00 @ baseRotationMatrix @
                                                                      dataAugmentationRotationMatrixX @
                                                                      dataAugmentationRotationMatrixY @
                                                                      dataAugmentationRotationMatrixZ)[0],
                                                   tvec=T_00, cameraMatrix=K_00, distCoeffs=D_00)

        # generate the image
        blank_image = np.zeros((int(cy * 2), int(cx * 2), 3), np.float32)
        for pixel, color in zip(imagePoints, out_colors):
            if ((int(pixel[0, 1]) < blank_image.shape[0]) and
                    (int(pixel[0, 0]) < blank_image.shape[1]) and
                    (int(pixel[0, 1]) > 0) and
                    (int(pixel[0, 0]) > 0)):
                blank_image[int(pixel[0, 1]), int(pixel[0, 0])] = color

        if ShowImage:
            plt.imshow(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR))
            plt.show()

        # return also the values drawn from the uniform distributions;
        sample = {'data': blank_image,
                  'label': sample['label'],
                  'random_Tx': random_Tx,
                  'random_Ty': random_Ty,
                  'random_Tz': random_Tz,
                  'random_Rx': random_Rx,
                  'random_Ry': random_Ry,
                  'random_Rz': random_Rz,
                  'starting_points': starting_points,
                  'remaining_points': remaining_points,
                  'generated_osm': np.zeros((int(cy * 2), int(cx * 2), 3), np.float32),
                  'negative_osm': np.zeros((int(cy * 2), int(cx * 2), 3), np.float32)
                  }

        if self.returnPoints:
            sample['save_out_points'] = save_out_points
            sample['save_out_colors'] = save_out_colors

        return sample
