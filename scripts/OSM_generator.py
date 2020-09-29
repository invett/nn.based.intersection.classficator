"""

THIS IS A TEST SCRIPT THAT ALLOWS US TO CREATE/VALIDATE THE CLASS THAT CREATES THE OSM-OG (OCCUPANCY GRID). THIS
CODE IS PART OF THE ICRA 2019 WORK BY CATTANEO D./BALLARDINI A.

PRIVATE ORIGINAL REPO can be found here:

ballardini@projects.ira.disco.unimib.it:/repository/git/road_intersection_detector.git

ballardini@johnny-i5:~/alvaro_ws/src/road_intersection_detector$ git log
commit f27412db2f17ee45a531cda69f5c11cb2f635271
Author: Daniele Cattaneo <daniele.cattaneo@disco.unimib.it>
Date:   Wed Feb 13 11:57:58 2019 +0100
On branch correlation_gpu

"""

from random import uniform

import argparse
from copy import copy, deepcopy
from math import cos, pi, sin, sqrt, fabs
from datetime import datetime
import numpy as np

import cv2

from miscellaneous.utils import bearing, degrees, radians, rotate_point, to_rotation_matrix_XYZRPY, npto_XYZRPY

from miscellaneous.utils import send_telegram_message
from miscellaneous.utils import send_telegram_picture
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Crossing Localization')

parser.add_argument('--std_rot', type=float, default=0.05, help='Std for arms rotation')
parser.add_argument('--max_width', type=float, default=6., help='MAX arms width')
parser.add_argument('--grid_test', type=float, default=None, nargs=3,
                    help='deltas for grid testing [delta_forward, delta_lateral, delta_rotation')

args, unknown = parser.parse_known_args()
print(args)


class Crossing:
    max_x = 30.
    max_y = 15.
    min_y = -15.
    gridCellSize = 0.1
    n_col = int(max_x / gridCellSize)
    n_row = int((max_y - min_y) / gridCellSize)

    def __init__(self, pose, num_arms, rotations_list, widths_list, center_global_xy):
        self.max_x = 30.
        self.max_y = 15.
        self.min_y = -15.
        self.gridCellSize = 0.1
        self.n_col = int(self.max_x / self.gridCellSize)
        self.n_row = int((self.max_y - self.min_y) / self.gridCellSize)
        self.pose = pose
        self.num_arms = num_arms
        self.real_rotation = deepcopy(rotations_list)
        self.rotations_list = rotations_list
        self.fixed_rotation_frame = deepcopy(
            self.rotations_list[0])  # This is used for limiting the arms rotation angle
        for i in range(num_arms):
            self.real_rotation[i] -= self.fixed_rotation_frame
        self.widths_list = widths_list
        self.rotation = 0.
        self.update_rotation()
        self.center_global_xy = center_global_xy
        self.update_rotation()
        self.center = [0.0, 0.0]
        self.update_center()

    def update_rotation(self):
        all_rotations = npto_XYZRPY(self.pose)
        self.rotation = all_rotations[5]

    def update_center(self):
        rotation = pi / 2 - self.rotation

        translation = [self.pose[0][3], self.pose[1][3]]
        distance_x = self.center_global_xy[0] - translation[0]
        distance_y = self.center_global_xy[1] - translation[1]
        center_pose = [distance_x * cos(rotation) - distance_y * sin(rotation) - self.min_y,
                       distance_y * cos(rotation) + distance_x * sin(rotation)]
        self.center = center_pose

    def generate_og(self, noise=True):
        return self.create_og_hypotesis(self.num_arms, self.rotations_list, self.widths_list, self.center,
                                        with_noise=noise)

    def rotate_point_og(self, p, center, angle):
        center_x = center[0]
        center_y = center[1]
        center_y = (self.max_x - center_y)
        return rotate_point(p, [center_x, center_y], angle)

    def distance(self, p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def convert_arc(self, pt1, pt2, sagitta, center):
        # extract point coordinates
        x1, y1 = pt1
        x2, y2 = pt2
        # if x1 < center:
        #    sagitta = -sagitta
        x1 = int(x1 / gridCellSize)
        x2 = int(x2 / gridCellSize)
        y1 = int(y1 / gridCellSize)
        y2 = int(y2 / gridCellSize)
        pt1 = [x1, y1]
        pt2 = [x2, y2]

        # find normal from midpoint, follow by length sagitta
        n = np.array([y2 - y1, x1 - x2])
        n_dist = np.sqrt(np.sum(n ** 2))

        if np.isclose(n_dist, 0):
            # catch error here, d(pt1, pt2) ~ 0
            print('Error: The distance between pt1 and pt2 is too small.')
            return (-1, -1), -1, -1, -1, (int(-1), int(-1))

        while sagitta > abs(n[0] / 8) or sagitta > abs(n[1] / 8):
            sagitta -= 1
        if sagitta < 1:
            sagitta = 1
        n = n / n_dist
        segno = +1
        x3, y3 = (np.array(pt1) + np.array(pt2)) / 2 + sagitta * n
        x4, y4 = (np.array(pt1) + np.array(pt2)) / 2 - sagitta * n
        if distance((x3, y3), center) > distance((x4, y4), center):
            segno = -1
            x3 = x4
            y3 = y4

        mean = (np.array(pt1) + np.array(pt2)) / 2
        # flood_x, flood_y = (np.array(pt1) + np.array(pt2))/2 + segno * (sagitta + 3) * n
        # flood_x, flood_y = mean[0] + 4*np.sign(n[0]), mean[1] + 4*np.sign(n[1])

        delta_x = float(mean[0] - center[0])
        delta_y = float(mean[1] - center[1])
        delta_point = 5
        if abs(delta_y / delta_x) < 0.1 or abs(delta_y / delta_x) > 100.:
            # print('Error: DELTA TOO HIGH OR TOO LOW')
            return (-1, -1), -1, -1, -1, (int(-1), int(-1))
        if mean[1] > center[1]:
            delta_y -= delta_point * abs(delta_y / delta_x)
        else:
            delta_y += delta_point * abs(delta_y / delta_x)

        if mean[0] > center[0]:
            delta_x -= delta_point
        else:
            delta_x += delta_point

        # print("DELTA: ",delta_y/delta_x)
        flood_x, flood_y = center[0] + delta_x, center[1] + delta_y

        # calculate the circle from three points
        # see https://math.stackexchange.com/a/1460096/246399
        A = np.array([[x1 ** 2 + y1 ** 2, x1, y1, 1], [x2 ** 2 + y2 ** 2, x2, y2, 1], [x3 ** 2 + y3 ** 2, x3, y3, 1]])
        M11 = np.linalg.det(A[:, (1, 2, 3)])
        M12 = np.linalg.det(A[:, (0, 2, 3)])
        M13 = np.linalg.det(A[:, (0, 1, 3)])
        M14 = np.linalg.det(A[:, (0, 1, 2)])

        if np.isclose(M11, 0):
            # catch error here, the points are collinear (sagitta ~ 0)
            print('Error: The third point is collinear.')
            return (-1, -1), -1, -1, -1, (int(-1), int(-1))

        cx = 0.5 * M12 / M11
        cy = -0.5 * M13 / M11
        radius = np.sqrt(cx ** 2 + cy ** 2 + M14 / M11)

        # calculate angles of pt1 and pt2 from center of circle
        pt1_angle = 180 * np.arctan2(y1 - cy, x1 - cx) / np.pi
        pt2_angle = 180 * np.arctan2(y2 - cy, x2 - cx) / np.pi

        return (cx, cy), radius, pt1_angle, pt2_angle, (int(flood_x), int(flood_y))

    def draw_ellipse(self, img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_AA,
                     shift=10):
        # uses the shift to accurately get sub-pixel resolution for arc
        # taken from https://stackoverflow.com/a/44892317/5087436
        center = (int(round(center[0] * 2 ** shift)), int(round(center[1] * 2 ** shift)))
        axes = (int(round(axes[0] * 2 ** shift)), int(round(axes[1] * 2 ** shift)))
        return cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)

    @staticmethod
    def add_noise(self, test, probability=1.0, elements_multiplier=1.0, distribution="normal"):
        """
            This is a static method that can be used even outside this class, as in teacher_tripletloss class inside
            sequencedataloader.py

        Args:
            self: itself...
            test: the image to add noise with
            probability: probability of being set as noise (for the pixel)
            elements_multiplier: the number of elements increases from bottom to top; this parameter increases this "ratio"
            distribution: "normal" or "uniform"; uniform seems to be more realistic

        Returns:
            same image with noise

        """
        # Old method with nested for cycles .. how to write a deamon!
        # tic = datetime.now()
        # for line in range(300, 0, -1):
        #
        #     num_elements = (300 - line) * elements_multiplier
        #
        #     if distribution == "normal":
        #         elements = np.trunc(np.random.normal(150.0, 50.0, int(num_elements))).tolist()  # i started with 10
        #     elif distribution == "uniform":
        #         elements = np.trunc(np.random.uniform(0.0, 300.0, int(num_elements))).tolist()  # i started with 10
        #     else:
        #         assert 1
        #
        #     for element in elements:
        #         if element > 299:
        #             element = 299
        #         if element < 0:
        #             element = 0
        #
        #         r = uniform(0.0, 1.0)
        #         t = probability  # (line / 300.)
        #
        #         if r < t:
        #             test[line-1, int(element)] = 0.0 # 255.0
        # toc = datetime.now()
        # delta = toc - tic
        # a = plt.figure()
        # plt.imshow(test)
        # send_telegram_picture(a, "OLD method: [sec:microsec] " + str(delta.seconds) + ":" + str(delta.microseconds))

        # tic = datetime.now()
        noise = np.ones((300, 300), test.dtype)
        for line in range(300, 0, -1):
            num_elements = (300 - line) * elements_multiplier
            # result = list(np.random.randint(0, 300, int(num_elements))) #takes much longer ... list+ randint++++
            result = (np.random.rand(1, int(num_elements)).squeeze() * 300.0).astype(np.int)
            if line != 300:
                noise[np.arange(noise.shape[0])[line, None], result] = 0

        if len(test.shape) == 3:  # check whether we're using a 3-channel image; in this case, 3-channelize the noise
            noise = np.dstack([noise] * 3)

        test = test * noise
        # toc = datetime.now()
        # delta = toc - tic
        # a = plt.figure()
        # plt.imshow(test)
        # send_telegram_picture(a, "NEW method: [sec:microsec] " + str(delta.seconds) + ":" + str(delta.microseconds))

        # third method, untested; this is slightly different from the original with the for-cycles and the replacement
        # noise = np.empty((0, 300), np.float32)
        # upperbound = 0.8
        # height = 300
        # normalizer = height / upperbound
        # for line in range(300, 0, -1):
        #     num_elements = (300 - line) / normalizer
        #     print(num_elements)
        #     line = np.random.uniform(0, 1, 300)
        #     check = line < num_elements
        #     line = np.where(check.all(), line, check)
        #     noise = np.append(noise, np.expand_dims(line, 0), axis=0)

        return test

    def create_og_hypotesis(self, howManyLanes, rotation_list, width_list, center=(14, 0), make_arcs=False,
                            with_noise=True):
        """

        Args:
            howManyLanes:   number of branches into the intersection
            rotation_list:  rotation of each of the intersections
            width_list:     widths of each of the incoming branch
            center:         where is the intersection with respect to the image/area
            make_arcs:      if true, between the arms we'll try to smooth the transition with arcs

        Returns:            numpy matrix containing the image in np.float32

        """

        # little check
        if len(rotation_list) != howManyLanes:
            return None

        crossing_image = np.zeros((self.n_col, self.n_row), np.float32)

        lines = np.zeros((howManyLanes, 2, 2), np.float32)  # magic numbers
        round_points = []  # for the arcs
        aaa = 3  # magic number
        max_width = np.array(width_list).max()
        center_x = center[0]
        center_y = center[1]
        center_image = (int(center[0] / self.gridCellSize), int((self.max_x - center_y) / self.gridCellSize))

        for i in range(howManyLanes):
            width = width_list[i]
            rotation = pi - rotation_list[i]

            # ROADS
            lines[i, 0, :] = [center_x, - self.max_x]
            lines[i, 1, :] = [center_x, self.max_x - center_y]

            # ARCS BETWEEN ROADS
            middle_point = [center_x, self.max_x - center_y]
            round_point1 = [middle_point[0] - width / 2, middle_point[1] - width / 2 - aaa]
            round_point2 = [middle_point[0] + width / 2, middle_point[1] - width / 2 - aaa]

            if rotation != 0.:
                lines[i, 0, :] = self.rotate_point_og(lines[i, 0, :], center, rotation)
                lines[i, 1, :] = self.rotate_point_og(lines[i, 1, :], center, rotation)

                # ARCS
                round_point1 = self.rotate_point_og(round_point1, center, rotation)
                round_point2 = self.rotate_point_og(round_point2, center,
                                                    rotation)  # round_point2 = (int(round_point1[0] / gridCellSize), int(round_point1[1] / gridCellSize))  # round_point1 = (int(round_point2[0] / gridCellSize), int(round_point2[1] / gridCellSize))

            round_points.append((round_point1, round_point2))

            lines[i] = lines[i] / self.gridCellSize
            cv2.line(crossing_image, tuple(lines[i, 0].astype(np.int32)), tuple(lines[i, 1].astype(np.int32)), 255,
                     int((width / self.gridCellSize)),
                     cv2.LINE_AA)  # cv2.fillPoly(crossing_image, [poly[i].astype(np.int32)], 1.)

        # THIS PART IS FOR THE "CURVES" BETWEEN TWO ROADS, THE LITTLE NICE ARCS
        if make_arcs:
            for i in range(howManyLanes):
                mask = np.zeros((302, 302), np.uint8)
                mask[:, :] = 255
                mask = cv2.circle(mask, center_image, int((max_width + aaa) / gridCellSize), 0, -1)
                for j in range(i, howManyLanes):
                    sagitta = 10
                    if i != j:
                        # print("real_pose_tf: ",int(width / gridCellSize))
                        # print("A: ",distance(round_points[i][0], round_points[j][0]))
                        # print("B: ",distance(round_points[i][0], round_points[j][1]))
                        # print("C: ",distance(round_points[i][1], round_points[j][0]))
                        # print("D: ",distance(round_points[i][1], round_points[j][1]))
                        try:
                            if self.distance(round_points[i][0], round_points[j][0]) < width:
                                center_arc, radius, start_angle, end_angle, flood_pt = self.convert_arc(
                                    round_points[i][0],
                                    round_points[j][0], sagitta,
                                    center_image)
                                if radius == -1:
                                    continue
                                axes = (radius, radius)
                                self.draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                                cv2.floodFill(crossing_image, mask, flood_pt, 255)
                            if self.distance(round_points[i][0], round_points[j][1]) < width:
                                center_arc, radius, start_angle, end_angle, flood_pt = self.convert_arc(
                                    round_points[i][0],
                                    round_points[j][1], sagitta,
                                    center_image)
                                if radius == -1:
                                    continue
                                axes = (radius, radius)
                                self.draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                                cv2.floodFill(crossing_image, mask, flood_pt, 255)
                            if self.distance(round_points[i][1], round_points[j][0]) < width:
                                center_arc, radius, start_angle, end_angle, flood_pt = self.convert_arc(
                                    round_points[i][1],
                                    round_points[j][0], sagitta,
                                    center_image)
                                if radius == -1:
                                    continue
                                axes = (radius, radius)
                                self.draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                                cv2.floodFill(crossing_image, mask, flood_pt, 255)
                            if self.distance(round_points[i][1], round_points[j][1]) < width:
                                center_arc, radius, start_angle, end_angle, flood_pt = self.convert_arc(
                                    round_points[i][1],
                                    round_points[j][1], sagitta,
                                    center_image)
                                if radius == -1:
                                    continue
                                axes = (radius, radius)
                                self.draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                                cv2.floodFill(crossing_image, mask, flood_pt, 255)
                        except:
                            print("CATCHATO QUALCOSA")
                            pass

        if with_noise:
            # crossing_image = self.add_noise(crossing_image, elements_multiplier=3., distribution="uniform") TODO check this out, there's an issue with the staticmethod
            crossing_image = self.add_noise(self=self, test=crossing_image, elements_multiplier=3.,
                                            distribution="uniform")

        return crossing_image


def test_crossing_pose(crossing_type=6, standard_width=6.0, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                       save=True, path="", filenumber=0, sampling=True, random_rate=1.0):
    """

    Args:
        crossing_type:   one of the 7 types of intersection
                    0 [ǁ]: straight
                    1 [⌜]: right turn
                    2 [⌝]: left turn
                    3 [⊣]: straight + right
                    4 [⊢]: straight + left
                    5 [⊤]: stop, left and right
                    6 [+]: full 4-ways intersection
        standard_width: standard width for intersection arms
        rnd_width: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)
        rnd_angle: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)
        rnd_spatial: parameter for uniform spatial cross position (center of the crossing area)
        noise: whether to add "noise" to the image or not. this wants to mimic the spatial "holes" in the BEV
        save: if true, save the image in the PATH parameter
        path: where to save the images
        filenumber: name of the file; please pass a number.
        sampling: whether or not add noise to intersection arms/branches; default True; set false to generate a
                  "canonical" intersection

    Returns: 300x300x3 uint8

    """

    # The center of the image is 15,0
    # full dimension = 30x30

    euler = np.array([0., 0., 0.])  # leave this fixed, is the rotation of the base centerpoint
    rotation = pi / 2 - euler[2]  # leave this fixed, is the rotation of the base centerpoint

    branches = 0
    rotation_list = []
    branch_widths = []

    xx = 15.0
    yy = 0.0

    rot_a = 0.0
    rot_b = 0.0
    rot_c = 0.0
    rot_d = 0.0

    width_a = 0.0
    width_b = 0.0
    width_c = 0.0
    width_d = 0.0

    # add noise to the sample; default behaviour
    if sampling:
        xx = 15.0 + uniform(-rnd_spatial, rnd_spatial) * random_rate
        yy = 0.0 + uniform(-rnd_spatial, rnd_spatial) * random_rate

        rot_a = uniform(-rnd_angle, rnd_angle) * random_rate
        rot_b = uniform(-rnd_angle, rnd_angle) * random_rate
        rot_c = uniform(-rnd_angle, rnd_angle) * random_rate
        rot_d = uniform(-rnd_angle, rnd_angle) * random_rate

        width_a = uniform(-rnd_width, rnd_width) * random_rate
        width_b = uniform(-rnd_width, rnd_width) * random_rate
        width_c = uniform(-rnd_width, rnd_width) * random_rate
        width_d = uniform(-rnd_width, rnd_width) * random_rate

    intersection_center = np.array([float(xx), float(yy), 0.])
    translation = np.array([0., 0., 0.])

    # distance_x = intersection_center[0] - translation[0]
    # distance_y = intersection_center[1] - translation[1]

    crossing_pose = to_rotation_matrix_XYZRPY(translation[0], translation[1], translation[2], euler[0], euler[1],
                                              euler[2])

    if crossing_type == 0:
        branches = 2
        rotation_list = [0. + rot_a, pi + rot_b]
        branch_widths = [6. + width_a, 6. + width_b]
    elif crossing_type == 1:
        branches = 2
        rotation_list = [0. + rot_a, pi / 2 + rot_b]
        branch_widths = [standard_width + width_a, standard_width + width_b]
    elif crossing_type == 2:
        branches = 2
        rotation_list = [0. + rot_a, 3 / 2 * pi + rot_b]
        branch_widths = [standard_width + width_a, standard_width + width_b]
    elif crossing_type == 3:
        branches = 3
        rotation_list = [0. + rot_a, pi + rot_c, 3 / 2 * pi + rot_d]
        branch_widths = [standard_width + width_a, standard_width + width_b, standard_width + width_c]
    elif crossing_type == 4:
        branches = 3
        rotation_list = [0. + rot_a, pi / 2 + rot_b, pi + rot_c]
        branch_widths = [standard_width + width_a, standard_width + width_b, standard_width + width_c]
    elif crossing_type == 5:
        branches = 3
        rotation_list = [0. + rot_a, pi / 2 + rot_b, 3 / 2 * pi + rot_d, ]
        branch_widths = [standard_width + width_a, standard_width + width_b, standard_width + width_c]
    elif crossing_type == 6:
        branches = 4
        rotation_list = [0. + rot_a, pi / 2 + rot_b, pi + rot_c, 3 / 2 * pi + rot_d]
        branch_widths = [standard_width + width_a, standard_width + width_b, standard_width + width_c,
                         standard_width + width_d]

    crossing_sample = Crossing(crossing_pose,  # matrix form... similar to intersection center, I won't change the code!
                               branches,  # how many arms in the intersection
                               rotation_list,  # rotation list (n# elements == n# arms)
                               branch_widths,  # width list (n# elements == n# arms)
                               (intersection_center[0], intersection_center[1])
                               )

    sample = crossing_sample.generate_og(noise=noise)

    if save:
        cv2.imwrite(str(path) + str(filenumber).zfill(10) + ".png", sample)

    # conversion to uint8 seems necessary for sending to telegram
    # and
    # create 3-channel image
    sample = np.dstack([np.array(sample / 1.0, dtype=np.uint8)] * 3)

    return [sample, xx, yy]


if __name__ == '__main__':

    i = 0
    for type in range(0, 7):
        for image in range(0, 100):
            sample = test_crossing_pose(crossing_type=type, path="/tmp/minchioline-sborrilla/",
                                        filenumber=i,
                                        noise=True,
                                        rnd_width=1.0,
                                        random_rate=1.0)
            i = i + 1
