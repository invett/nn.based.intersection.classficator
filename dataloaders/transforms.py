import cv2.cv2 as cv2
import kornia
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from skimage.transform import resize

from miscellaneous.utils import getCameraRototraslation, radians

# For debugging
ShowImage = False

if ShowImage:
    import matplotlib.pyplot as plt


class GenerateNewDataset(object):
    """
    This simply sets a value that will be used in the getitem to save the image (with the correct filename) inside the
    path passed as parameter
    """

    def __init__(self, path, save=True):
        self.path = path
        self.save = save  # this is used to disable this 'fake-transform' even if present...

    def __call__(self, sample):
        if self.save:
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
        if 'generated_osm' in sample:
            osm = sample['generated_osm']
        if 'negative_osm' in sample:
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
        if 'generated_osm' in sample:
            osm = resize(osm, (new_h, new_w), anti_aliasing=True)
        if 'negative_osm' in sample:
            osm_neg = resize(osm_neg, (new_h, new_w), anti_aliasing=True)

        sample['data'] = image
        if 'generated_osm' in sample:
            sample['generated_osm'] = osm
        if 'negative_osm' in sample:
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
    """Convert nd arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['data'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # generated_osm might be optional, let's handle this
        if 'generated_osm' in sample:
            generated_osm = sample['generated_osm']
            generated_osm = generated_osm.transpose((2, 0, 1))
            if 'negative_osm' in sample:
                negative_osm = sample['negative_osm']
                negative_osm = negative_osm.transpose((2, 0, 1))
                return {'data': torch.from_numpy(image),
                        'generated_osm': torch.from_numpy(generated_osm).float(),
                        'negative_osm': torch.from_numpy(negative_osm).float(),
                        'neg_label': sample['neg_label'],
                        'label': label,
                        'image_path': sample['image_path']}
            else:
                return {'data': torch.from_numpy(image),
                        'generated_osm': torch.from_numpy(generated_osm).float(),
                        'label': label,
                        'neg_label': sample['neg_label'],
                        'image_path': sample['image_path']}
        else:
            return {'data': torch.from_numpy(image),
                    'label': label,
                    'neg_label': sample['neg_label'],
                    'image_path': sample['image_path']}


class Normalize(object):
    """
    Converts the range from 0..255 >> 0..1 (just to be used inside Pytorch)
    OPTIMIZE this might be faster directly in GPU/Tensor...
    """

    def __call__(self, sample):
        image_norm = cv2.normalize(sample['data'], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
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


class GenerateWarping(object):

    """
        Creates the image-warping from RGB to Bird Eye View with no mask or other amenities

        WARNING: all the hard-coded values, height and position of camera etc. are tested with
        400,400 output size to be compatible with the OSM generator and all the previous code
        
    """

    def __init__(self, cameramatrix=None,  bev_height=400, bev_width=400,
                 random_Rx_degrees=0.0,
                 random_Ry_degrees=0.0,
                 random_Rz_degrees=0.0,
                 random_Tx_meters=0.0,
                 random_Ty_meters=0.0,
                 random_Tz_meters=0.0,
                 warpdataset='kitti',
                 ignoreAllGivenRandomValues=False):
        #######################################################################################
        # Code for the warping of RGB images - we use KORNIA to get the perspective transform #
        #######################################################################################

        # ordering of points:
        #
        #        BEV                                 3D
        #
        #     3----<----2                     3-------<-------2
        #               |                    /                 \
        #               ^                   /                   \
        #               |                  /                     \
        #     0---->----1                 0----------->-----------1
        #

        # KITTI camera matrix handle
        if cameramatrix is None:
            if warpdataset == 'kitti' or warpdataset == 'KITTI-ROAD-WARPING':
                self.K = np.array([[9.786977e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00],
                                   [0.000000e+00, 9.717435e+02, 2.497222e+02, 0.000000e+00],
                                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]], dtype=np.float64)

            elif warpdataset == 'kitti360':
                # kitti 360 P_rect_00
                self.K = np.array([[552.554261, 0.00000000, 682.049453, 0.000000],
                                   [0.00000000, 552.554261, 238.769549, 0.000000],
                                   [0.00000000, 0.00000000, 1.00000000, 0.000000]], dtype=np.float64)

            elif (warpdataset == 'alcala26012021') or (warpdataset == 'alcala26012021') or (
                    warpdataset == 'alcala-12.02.2021.000') or warpdataset == 'alcala-12.02.2021.001':
                # sj7 star
                self.K = np.array([[300.000000, 0.00000000, 800.0 / 2., 0.000000],
                                   [0.00000000, 300.000000, 450.0 / 2., 0.000000],
                                   [0.00000000, 0.00000000, 1.00000000, 0.000000]], dtype=np.float64)
        else:
            self.K = cameramatrix

        self.random_Rx_radians = radians(random_Rx_degrees)
        self.random_Ry_radians = radians(random_Ry_degrees)
        self.random_Rz_radians = radians(random_Rz_degrees)
        self.random_Tx_meters = random_Tx_meters
        self.random_Ty_meters = random_Ty_meters
        self.random_Tz_meters = random_Tz_meters
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.warpdataset = warpdataset

        assert self.random_Rx_radians is not None, "transform value can't be None in GenerateWarping"
        assert self.random_Ry_radians is not None, "transform value can't be None in GenerateWarping"
        assert self.random_Rz_radians is not None, "transform value can't be None in GenerateWarping"
        assert self.random_Tx_meters is not None, "transform value can't be None in GenerateWarping"
        assert self.random_Ty_meters is not None, "transform value can't be None in GenerateWarping"
        assert self.random_Tz_meters is not None, "transform value can't be None in GenerateWarping"

        self.ignoreAllGivenRandomValues = ignoreAllGivenRandomValues

    def __call__(self, sample):
        """

        Args:
            sample: the sample from the dataloader

        Returns: the bird eye view on ['data']

        """

        if self.ignoreAllGivenRandomValues:
            random_Rx = 0.0
            random_Ry = 0.0
            random_Rz = 0.0
            random_Tx = 0.0
            random_Ty = 0.0
            random_Tz = 0.0
        else:
            random_Rx = np.random.uniform(-self.random_Rx_radians, self.random_Rx_radians)
            random_Ry = np.random.uniform(-self.random_Ry_radians, self.random_Ry_radians)
            random_Rz = np.random.uniform(-self.random_Rz_radians, self.random_Rz_radians)
            random_Tx = np.random.uniform(0.0, self.random_Tx_meters)
            random_Ty = np.random.uniform(-self.random_Ty_meters, self.random_Ty_meters)
            random_Tz = np.random.uniform(-self.random_Tz_meters, self.random_Tz_meters)

        if self.warpdataset == 'KITTI-ROAD-WARPING':
            ## don't know why the other 'kitti' does not work .. lets redo this here
            dx = 8 + random_Tx
            dy = 0 + random_Ty
            dz = 2.0 + random_Tz
            pitchCorrection = 0.084 + random_Rx
            yawCorrection = 0.09 + random_Ry
            rollCorrection = 0.0 + random_Rz
            # the importan one here are:
            # first [x x x x] [y y y y] [....] [....]
            points_3d = np.array([[16, 16, 80, 80], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        elif self.warpdataset == 'kitti':
            # position of the virtual camera -- standard kitti
            dx = 6 + random_Tx
            dy = 0 + random_Ty
            dz = 2.0 + random_Tz
            pitchCorrection = 0.084 + random_Rx
            yawCorrection = 0.1 + random_Ry
            rollCorrection = 0.0 + random_Rz
            # the importan one here are:              
            # first [x x x x] [y y y y] [....] [....]
            points_3d = np.array([[16, 16, 120, 120], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        elif self.warpdataset == 'kitti360':
            # position of the virtual camera -- standard kitti 360
            dx = 11   + random_Tx
            dy = 0    + random_Ty
            dz = 2.0  + random_Tz
            pitchCorrection = 0.11   + random_Rx  # 0.29 deg 0.005rad feasible
            yawCorrection = -0.085  + random_Ry
            rollCorrection = 0.000  + random_Rz
            # the importan one here are:
            # first [x x x x] [y y y y] [....] [....]
            #old --- points_3d = np.array([[16, 16, 120, 120], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
            points_3d = np.array([[16, 16, 80, 80], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        elif self.warpdataset == 'alcala26012021':
            # position of the virtual camera -- standard kitti 360
            dx = 12   + random_Tx  # position of the camera: distance
            dy = 1    + random_Ty  # position of the camera: positive, to the left
            dz = 1.5  + random_Tz  # height of the camera; positive, more up :)
            pitchCorrection = 0.008  + random_Rx  # 0.29 deg 0.005rad feasible
            yawCorrection =   0.14   + random_Ry
            rollCorrection =  0.000  + random_Rz
            # the importan one here are:
            # first [x x x x] [y y y y] [....] [....]
            # points_3d = np.array([[13, 13, 120, 120], [26, -26, -26, 26], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
            points_3d = np.array([[14, 14, 60, 60], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        elif self.warpdataset == 'alcala-12.02.2021.000':
            # position of the virtual camera
            # ALCALA DATASET OF 12.02.2021, NOON, WITH AUGUSTO'S FORD FOCUS
            dx = 12   + random_Tx  # position of the camera: distance
            dy = 1    + random_Ty  # position of the camera: positive, to the left
            dz = 1.5  + random_Tz  # height of the camera; positive, more up :)
            pitchCorrection =  -0.028 + random_Rx  # 0.29 deg 0.005rad feasible
            yawCorrection   =  -0.03  + random_Ry
            rollCorrection  =   0.00  + random_Rz
            # the importan one here are:
            # first [x x x x] [y y y y] [....] [....]
            # points_3d = np.array([[13, 13, 120, 120], [26, -26, -26, 26], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
            points_3d = np.array([[14, 14, 60, 60], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        elif self.warpdataset == 'alcala-12.02.2021.001':
            # position of the virtual camera
            # ALCALA DATASET OF 12.02.2021, AFTERNOON, WITH THE C4
            dx = 12   + random_Tx  # position of the camera: distance
            dy = 1    + random_Ty  # position of the camera: positive, to the left
            dz = 1.5  + random_Tz  # height of the camera; positive, more up :)
            pitchCorrection = 0.0925 + random_Rx  # 0.29 deg 0.005rad feasible
            yawCorrection =   -0.09  + random_Ry
            rollCorrection =  0.000  + random_Rz
            # the importan one here are:
            # first [x x x x] [y y y y] [....] [....]
            # points_3d = np.array([[14, 14, 120, 120], [26, -26, -26, 26], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
            points_3d = np.array([[14, 14, 60, 60], [16, -16, -16, 16], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float64)
        else:
            assert 0, "unknown warping .... check generatewarping"

        points_dst = torch.FloatTensor(
            [[[0, self.bev_width], [self.bev_height, self.bev_width], [self.bev_height, 0], [0, 0], ]])
        WorldToCam = np.linalg.inv(getCameraRototraslation(pitchCorrection, yawCorrection, rollCorrection, dx, dy, dz))
        points_2d = self.K @ WorldToCam @ points_3d
        points_2d = points_2d[:, :] / points_2d[2, :]
        points_2d = points_2d[:2, :]
        self.M = kornia.get_perspective_transform(
            torch.tensor(np.expand_dims(np.transpose(np.asarray(points_2d, dtype=np.float32)), axis=0)), points_dst)

        img = kornia.image_to_tensor(sample['image_02'], keepdim=False)

        data_warp = kornia.warp_perspective(img.float(), self.M, dsize=(self.bev_height, self.bev_width))
        warped = kornia.tensor_to_image(data_warp.byte())
        warped = np.asarray(warped, dtype=np.float32)

        sample = {'data': warped,  # [400, 400, 3] to be compatible
                  'label': sample['label'],
                  'random_Tx': random_Tx,
                  'random_Ty': random_Ty,
                  'random_Tz': random_Tz,
                  'random_Rx': random_Rx,
                  'random_Ry': random_Ry,
                  'random_Rz': random_Rz,
                  'starting_points': 0,
                  'remaining_points': 0,
                  'generated_osm': np.zeros((int(self.bev_height * 2), int(self.bev_width * 2), 3), np.float32),
                  'negative_osm': np.zeros((int(self.bev_height * 2), int(self.bev_width * 2), 3), np.float32)
                  }

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
                 max_front_distance=50.0,
                 max_height=3.0,
                 decimate=1.0,
                 random_Rx_degrees=2.0,
                 random_Ry_degrees=15.0,
                 random_Rz_degrees=2.0,
                 random_Tx_meters=2.0,
                 random_Ty_meters=2.0,
                 random_Tz_meters=2.0,
                 base_Tx=0.000000e+00,
                 base_Ty=17.00000e+00,
                 base_Tz=10.50000e+00,
                 returnPoints=False,
                 excludeMask=False,
                 qmatrix='kitti'
                 ):
        self.max_front_distance = max_front_distance
        self.max_height = max_height
        self.decimate = decimate
        self.random_Rx_degrees = random_Rx_degrees
        self.random_Ry_degrees = random_Ry_degrees
        self.random_Rz_degrees = random_Rz_degrees
        self.random_Tx_meters = random_Tx_meters
        self.random_Ty_meters = random_Ty_meters
        self.random_Tz_meters = random_Tz_meters
        self.returnPoints = returnPoints
        self.excludeMask = excludeMask
        self.qmatrix = qmatrix

        self.base_Tx = base_Tx
        self.base_Ty = base_Ty
        self.base_Tz = base_Tz

        assert self.random_Rx_degrees is not None, "transform value can't be None in GenerateBEV"
        assert self.random_Ry_degrees is not None, "transform value can't be None in GenerateBEV"
        assert self.random_Rz_degrees is not None, "transform value can't be None in GenerateBEV"
        assert self.random_Tx_meters is not None, "transform value can't be None in GenerateBEV"
        assert self.random_Ty_meters is not None, "transform value can't be None in GenerateBEV"
        assert self.random_Tz_meters is not None, "transform value can't be None in GenerateBEV"

    def __call__(self, sample):

        # this Q matrix was obtained using STEREORECTIFY; would be nice to import this part of the code too.
        # in the meanwhile, check the wiki page with a mini-tutorial
        if self.qmatrix == 'kitti':
            rev_proj_matrix = np.array([
                [1., 0., 0., -607.19281006],
                [0., 1., 0., -185.21570587],
                [0., 0., 0., 718.85601807],
                [0., 0., -1.85185185, 0.]], dtype=np.float64)

        elif self.qmatrix == 'kitti360':
            # KITTI-360
            rev_proj_matrix = np.array([
                 [1., 0.,  0., -682.04943848],
                 [0., 1.,  0., -238.76953888],
                 [0., 0.,  0.,  552.55426025],
                 [0., 0., -1.66666667, 0.]], dtype=np.float64)
        else:
            print("Invalid qmatrix parameter")
            exit(-1)

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
            idx = np.fabs(save_out_points[:, :, 2]) < self.max_front_distance
            save_out_points = save_out_points[idx]
            save_out_colors = save_out_colors
            save_out_colors = save_out_colors[idx]

        # ALVARO MASK # TODO this is the right place to disable ALVARO MASK s and so get the FULL - BEVs
        alvaro = sample['alvaromask']

        # this if was added once we had kitti360
        if alvaro is not None:
            # nice trick to avoid touching more code than needed... from this out_points needs to be 453620 x 3
            if self.excludeMask:
                alvaro = np.ones(alvaro.shape, dtype=alvaro.dtype)
            out_points = out_points[alvaro > 0]
            out_colors = out_colors[alvaro > 0]
        else:
            out_points = out_points.reshape([-1, 3])
            out_colors = out_colors.reshape([-1, 3])


        # filter by dimension
        # idx = np.fabs(out_points[:, 2]) < self.max_front_distance
        # out_points = out_points[idx]
        # out_colors = out_colors.reshape(-1, 3)
        # out_colors = out_colors[idx]

        # filter by dimension : "front" first, then "height"
        idx = np.fabs(out_points[:, 2]) < self.max_front_distance
        out_points = out_points[idx]
        out_colors = out_colors[idx]
        idx = np.fabs(out_points[:, 1]) < self.max_height
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

        T_00 = np.array([self.base_Tx + random_Tx,
                         self.base_Ty + random_Ty,
                         self.base_Tz + random_Tz], dtype=np.float64)

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

        # avoid performing rotations of type:1 and type:2 intersections as the should appear as type:0
        if sample['label'] == 1 or sample['label'] == 2:
            random_Ry = 0.0

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
