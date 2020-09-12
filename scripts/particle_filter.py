
from random import uniform

import argparse
from copy import copy, deepcopy
from math import cos, pi, sin, sqrt, fabs

import numpy as np

import cv2

from miscellaneous.utils import bearing, degrees, radians, rotate_point, to_rotation_matrix_XYZRPY, npto_XYZRPY

parser = argparse.ArgumentParser(description='Crossing Localization')

parser.add_argument('--std_rot', type=float, default=0.05, help='Std for arms rotation')
parser.add_argument('--max_width', type=float, default=6., help='MAX arms width')
parser.add_argument('--grid_test', type=float, default=None, nargs=3, help='deltas for grid testing [delta_forward, delta_lateral, delta_rotation')

args, unknown = parser.parse_known_args()
print(args)


class Crossing:
    max_x = 30.
    max_y = 15.
    min_y = -15.
    gridCellSize = 0.1
    n_col = int(max_x / gridCellSize)
    n_row = int((max_y - min_y) / gridCellSize)
    street_distribution_sigma = 0.1
    angle_distribution_sigma = 0.1
    street_distribution_alpha = 0.5
    angle_distribution_alpha = 0.5

    def __init__(self, pose, num_arms, rotations_list, widths_list, center_global_xy):
        self.pose = pose
        self.num_arms = num_arms
        self.real_rotation = deepcopy(rotations_list)
        self.rotations_list = rotations_list
        self.fixed_rotation_frame = deepcopy(self.rotations_list[0])  # This is used for limiting the arms rotation angle
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
        center_pose = [distance_x * cos(rotation) - distance_y * sin(rotation) - min_y,
                       distance_y * cos(rotation) + distance_x * sin(rotation)]
        self.center = center_pose

    def generate_og(self):
        return create_og_hypotesis(self.num_arms, self.rotations_list, self.widths_list, self.center)


max_x = 30.
max_y = 15.
min_y = -15.
gridCellSize = 0.1
n_col = int(max_x / gridCellSize)
n_row = int((max_y - min_y) / gridCellSize)


def rotate_point_og(p, center, angle):
    center_x = center[0]
    center_y = center[1]
    center_y = (max_x - center_y)
    return rotate_point(p, [center_x, center_y], angle)


def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def convert_arc(pt1, pt2, sagitta, center):
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


def draw_ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
    # taken from https://stackoverflow.com/a/44892317/5087436
    center = (int(round(center[0] * 2 ** shift)), int(round(center[1] * 2 ** shift)))
    axes = (int(round(axes[0] * 2 ** shift)), int(round(axes[1] * 2 ** shift)))
    return cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)


def add_noise(test, probability=1.0, elements_multiplier=1.0, distribution="normal"):
    """

    Args:
        test: the image to add noise with
        probability: probability of being set as noise (for the pixel)
        elements_multiplier: the number of elements increases from bottom to top; this parameter increases this "ratio"
        distribution: "normal" or "uniform"; uniform seems to be more realistic

    Returns:
        same image with noise

    """
    for line in range(300, 0, -1):

        num_elements = (300 - line) * elements_multiplier

        if distribution == "normal":
            elements = np.trunc(np.random.normal(150.0, 50.0, int(num_elements))).tolist()  # i started with 10
        elif distribution == "uniform":
            elements = np.trunc(np.random.uniform(0.0, 300.0, int(num_elements))).tolist()  # i started with 10
        else:
            assert 1

        for element in elements:
            if element > 299:
                element = 299
            if element < 0:
                element = 0

            r = uniform(0.0, 1.0)
            t = probability  # (line / 300.)

            if r < t:
                test[line-1, int(element)] = 0.0 # 255.0

    return test


def create_og_hypotesis(howManyLanes, rotation_list, width_list, center=(14, 0), make_arcs=False, with_noise=True):
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

    crossing_image = np.zeros((n_col, n_row), np.float32)

    lines = np.zeros((howManyLanes, 2, 2), np.float32)                                      # magic numbers
    round_points = []                                                                       # for the arcs
    aaa = 3                                                                                 # magic number
    max_width = np.array(width_list).max()
    center_x = center[0]
    center_y = center[1]
    center_image = (int(center[0] / gridCellSize), int((max_x - center_y) / gridCellSize))

    for i in range(howManyLanes):
        width = width_list[i]
        rotation = pi - rotation_list[i]

        # ROADS
        lines[i, 0, :] = [center_x, -max_x]
        lines[i, 1, :] = [center_x, max_x - center_y]

        # ARCS BETWEEN ROADS
        middle_point = [center_x, max_x - center_y]
        round_point1 = [middle_point[0] - width / 2, middle_point[1] - width / 2 - aaa]
        round_point2 = [middle_point[0] + width / 2, middle_point[1] - width / 2 - aaa]

        if rotation != 0.:
            lines[i, 0, :] = rotate_point_og(lines[i, 0, :], center, rotation)
            lines[i, 1, :] = rotate_point_og(lines[i, 1, :], center, rotation)

            #ARCS
            round_point1 = rotate_point_og(round_point1, center, rotation)
            round_point2 = rotate_point_og(round_point2, center,
                                           rotation)  # round_point2 = (int(round_point1[0] / gridCellSize), int(round_point1[1] / gridCellSize))  # round_point1 = (int(round_point2[0] / gridCellSize), int(round_point2[1] / gridCellSize))

        round_points.append((round_point1, round_point2))

        # poly[i] = poly[i] / gridCellSize
        lines[i] = lines[i] / gridCellSize
        cv2.line(crossing_image, tuple(lines[i, 0].astype(np.int32)), tuple(lines[i, 1].astype(np.int32)), 255,
                 int((width / gridCellSize)), cv2.LINE_AA)  # cv2.fillPoly(crossing_image, [poly[i].astype(np.int32)], 1.)

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
                        if distance(round_points[i][0], round_points[j][0]) < width:
                            center_arc, radius, start_angle, end_angle, flood_pt = convert_arc(round_points[i][0],
                                                                                               round_points[j][0], sagitta,
                                                                                               center_image)
                            if radius == -1:
                                continue
                            axes = (radius, radius)
                            draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                            cv2.floodFill(crossing_image, mask, flood_pt, 255)
                        if distance(round_points[i][0], round_points[j][1]) < width:
                            center_arc, radius, start_angle, end_angle, flood_pt = convert_arc(round_points[i][0],
                                                                                               round_points[j][1], sagitta,
                                                                                               center_image)
                            if radius == -1:
                                continue
                            axes = (radius, radius)
                            draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                            cv2.floodFill(crossing_image, mask, flood_pt, 255)
                        if distance(round_points[i][1], round_points[j][0]) < width:
                            center_arc, radius, start_angle, end_angle, flood_pt = convert_arc(round_points[i][1],
                                                                                               round_points[j][0], sagitta,
                                                                                               center_image)
                            if radius == -1:
                                continue
                            axes = (radius, radius)
                            draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                            cv2.floodFill(crossing_image, mask, flood_pt, 255)
                        if distance(round_points[i][1], round_points[j][1]) < width:
                            center_arc, radius, start_angle, end_angle, flood_pt = convert_arc(round_points[i][1],
                                                                                               round_points[j][1], sagitta,
                                                                                               center_image)
                            if radius == -1:
                                continue
                            axes = (radius, radius)
                            draw_ellipse(crossing_image, center_arc, axes, 0, start_angle, end_angle, 255)
                            cv2.floodFill(crossing_image, mask, flood_pt, 255)
                    except:
                        print("CATCHATO QUALCOSA")
                        pass

    if with_noise:
        crossing_image = add_noise(crossing_image, elements_multiplier=3., distribution="uniform")

    return crossing_image


def test_crossing_pose(crossing_type=6, standard_width=6.0, rnd_width=2.0, rnd_angle=0.4, noise=True, save=True,
                       path="", filenumber=0):
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
        noise: whether to add "noise" to the image or not. this wants to mimic the spatial "holes" in the BEV
        save: if true, save the image in the PATH parameter
        path: where to save the images
        filenumber: name of the file; please pass a number.

    Returns:

    """

    # The center of the image is 15,0
    # full dimension = 30x30

    euler = np.array([0., 0., 0.])  # leave this fixed, is the rotation of the base centerpoint
    rotation = pi / 2 - euler[2]  # leave this fixed, is the rotation of the base centerpoint

    branches = 0
    rotation_list = []
    branch_widths = []

    xx = 15.0 + uniform(-9., 9.)
    yy = 0.0 + uniform(-9., 9.)

    rot_a = uniform(-rnd_angle, rnd_angle)
    rot_b = uniform(-rnd_angle, rnd_angle)
    rot_c = uniform(-rnd_angle, rnd_angle)
    rot_d = uniform(-rnd_angle, rnd_angle)

    width_a = uniform(-rnd_width, rnd_width)
    width_b = uniform(-rnd_width, rnd_width)
    width_c = uniform(-rnd_width, rnd_width)
    width_d = uniform(-rnd_width, rnd_width)

    intersection_center = np.array([float(xx), float(yy), 0.])
    translation = np.array([0., 0., 0.])

    distance_x = intersection_center[0] - translation[0]
    distance_y = intersection_center[1] - translation[1]

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
        rotation_list = [0. + rot_a,  pi + rot_c, 3 / 2 * pi + rot_d]
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
        branch_widths = [standard_width + width_a, standard_width + width_b, standard_width + width_c, standard_width + width_d]

    crossing_sample = Crossing(crossing_pose,  # matrix form... similar to intersection center, I won't change the code!
                               branches,       # how many arms in the intersection
                               rotation_list,  # rotation list (n# elements == n# arms)
                               branch_widths,  # width list (n# elements == n# arms)
                               (intersection_center[0], intersection_center[1])
                               )

    sample = crossing_sample.generate_og()

    if save:
        cv2.imwrite(str(path) + str(filenumber).zfill(10) + ".png", sample)

    if False:

        counter = 0

        for i in range(360):

            xx = 15.0 + uniform(-9., 9.)
            yy = 0.0  + uniform(-9., 9.)
            rot_a = uniform(-0.4, 0.4)
            rot_b = uniform(-0.4, 0.4)
            rot_c = uniform(-0.4, 0.4)
            rot_d = uniform(-0.4, 0.4)

            width_a = uniform(-2.0, 2.0)
            width_b = uniform(-2.0, 2.0)
            width_c = uniform(-2.0, 2.0)
            width_d = uniform(-2.0, 2.0)


            counter = 0
            intersection_center = np.array([float(xx), float(yy), 0.])
            translation = np.array([0., 0., 0.])

            distance_x = intersection_center[0] - translation[0]
            distance_y = intersection_center[1] - translation[1]
            center_pose = [distance_x * cos(rotation) - distance_y * sin(rotation) - min_y,
                           distance_y * cos(rotation) + distance_x * sin(rotation)]

            crossing_pose = to_rotation_matrix_XYZRPY(translation[0], translation[1], translation[2], euler[0], euler[1],
                                                      euler[2])

            crossing_sample = Crossing(copy(crossing_pose),
                                       4,                   # how many arms in the intersection
                                       copy([0. + rot_a,
                                             pi / 2 + rot_b,
                                             pi + rot_c,
                                             3/2 * pi + rot_d,
                                             ]),  # rotation list (n# elements == n# arms)
                                       copy([6. + width_a,
                                             6. + width_b,
                                             6. + width_c,
                                             6. + width_d
                                             ]),  # width list (n# elements == n# arms)
                                       (intersection_center[0], intersection_center[1])  # global center
                                       )

            og = crossing_sample.generate_og()

            cv2.imwrite("/tmp/minchioline-sborrilla/" + str(i).zfill(4) + ".png", og)

    # here i was trying to understand what the EULER+ROTATION was; => DON'T TOUCH "euler"
    if False:

        xx = 15.0
        yy = 0.0

        counter = 0
        intersection_center = np.array([float(xx), float(yy), 0.])
        translation = np.array([0., 0., 0.])

        base_rotation = 0.0
        for i in range(360):

            distance_x = intersection_center[0] - translation[0]
            distance_y = intersection_center[1] - translation[1]
            center_pose = [distance_x * cos(rotation) - distance_y * sin(rotation) - min_y,
                           distance_y * cos(rotation) + distance_x * sin(rotation)]


            crossing_pose = to_rotation_matrix_XYZRPY(translation[0], translation[1], translation[2], euler[0], euler[1], euler[2])

            crossing_sample = Crossing(copy(crossing_pose),
                                       3,
                                       copy([0.+base_rotation, pi / 2 , pi]),        # rotation list
                                       copy([6., 6., 6.]),            # width list
                                       (intersection_center[0], intersection_center[1]) # global center
                                       )

            og = crossing_sample.generate_og()

            cv2.imwrite("/tmp/minchioline-sborrilla/" + str(counter).zfill(4) + ".png", og)
            counter = counter + 1
            base_rotation = base_rotation + radians(1.0)

    # moving aroung the position => USE CROSS_POSE
    if False:
        counter = 0
        for xx in range(0, 30):
            for yy in range(-15, 15):
                intersection_center = np.array([float(xx), float(yy), 0.])
                translation = np.array([0., 0., 0.])

                euler = np.array([0., 0., 0.0])  #leave this fixed, is the rotatino of the base centerpoint
                rotation = pi / 2 - euler[2]     #leave this fixed, is the rotatino of the base centerpoint

                distance_x = intersection_center[0] - translation[0]
                distance_y = intersection_center[1] - translation[1]
                center_pose = [distance_x * cos(rotation) - distance_y * sin(rotation) - min_y,
                               distance_y * cos(rotation) + distance_x * sin(rotation)]


                crossing_pose = to_rotation_matrix_XYZRPY(translation[0], translation[1], translation[2], euler[0], euler[1], euler[2])

                crossing_sample = Crossing(copy(crossing_pose),
                                           3,
                                           copy([0., pi / 2 - 0.4, pi]),        # rotation list
                                           copy([6., 6., 6.]),            # width list
                                           (intersection_center[0], intersection_center[1]) # global center
                                           )

                og = crossing_sample.generate_og()

                cv2.imwrite("/tmp/minchioline-sborrilla/" + str(counter).zfill(4) + ".png", og)
                counter = counter + 1


if __name__ == '__main__':

    for i in range(0, 10):
        test_crossing_pose(crossing_type=6, path="/tmp/minchioline-sborrilla/", filenumber=i)
