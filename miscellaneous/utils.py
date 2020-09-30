import numpy as np
import requests
import json
import datetime

from io import BytesIO
import linecache
import sys

from math import pi, cos, sin, atan2, asin
from functools import reduce

import torch
from torch import nn
from sklearn.metrics import accuracy_score

import random


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


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


telegram_token = "1178257144:AAH5DEYxJjPb0Qm_afbGTuJZ0-oqfIMFlmY"  # replace TOKEN with your bot's token
telegram_channel = '-1001352516993'


def send_telegram_message(message):
    """

    Args:
        message: text

    Returns: True if ok

    """
    URI = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + telegram_channel + '&parse_mode=Markdown&text=' + str(
        datetime.datetime.now()) + "\n" + message
    response = requests.get(URI)
    return json.loads(response.content)['ok']


def send_telegram_picture(plt, description):
    """

    Args:
        plt: matplotlib.pyplot
        description: sends a figure with the confusion matrix through the telegram channel

    Returns: True if ok

    """
    figdata = BytesIO()
    plt.savefig(figdata, format='png')
    URI = 'https://api.telegram.org/bot' + telegram_token + '/sendPhoto?chat_id=' + telegram_channel + "&caption=" + str(
        datetime.datetime.now()) + "\n" + description
    pic = {'photo': ("Foto", figdata.getvalue(), 'image/png')}
    response = requests.get(URI, files=pic)

    return json.loads(response.content)['ok']


def euler2mat(z, y, x):
    """

    Args:
        z: Yaw
        y: Pitch
        x: Roll

    Returns:
        rotation matrix as Matlab

        matrix = eul2tform([YAW PITCH ROLL], 'XYZ')

    """
    Ms = []
    if z:
        cosz = cos(z)
        sinz = sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0, 0],
             [sinz, cosz, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]
             ]))
    if y:
        cosy = cos(y)
        siny = sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny, 0],
             [0, 1, 0, 0],
             [-siny, 0, cosy, 0],
             [0, 0, 0, 1]
             ]))
    if x:
        cosx = cos(x)
        sinx = sin(x)
        Ms.append(np.array(
            [[1, 0, 0, 0],
             [0, cosx, -sinx, 0],
             [0, sinx, cosx, 0],
             [0, 0, 0, 1]
             ]))
    if Ms:
        return reduce(np.dot, Ms[::-1])  # equivale a Ms[2]@Ms[1]@Ms[0]

    # nel caso sfigato, restituiscimi una idenatità (era 3x3, diventa 4x4)
    return np.eye(4)


def npxyz2mat(x, y, z):
    # todo TRANSFORM TO NUMPY --  assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = np.eye(4)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat


def to_rotation_matrix_XYZRPY(x, y, z, roll, pitch, yaw):
    R = euler2mat(yaw, pitch,
                  roll)  # la matrice che viene fuori corrisponde a eul2tform di matlab (Convert Euler angles to homogeneous transformation)
    T = npxyz2mat(x, y, z)
    RT = np.matmul(T, R)
    return RT


def npto_XYZRPY(rotmatrix):
    '''
    Usa mathutils per trasformare una matrice di trasformazione omogenea in xyzrpy
    https://docs.blender.org/api/master/mathutils.html#
    WARNING: funziona in 32bits quando le variabili numpy sono a 64 bit

    :param rotmatrix: np array
    :return: np array with the xyzrpy
    '''

    #### TODO DELETE -->> "old version" ---> mat = mathutils.Matrix()
    #### TODO DELETE -->> "old version" ---> mat[0][0:4] = rotmatrix[0][0],rotmatrix[0][1],rotmatrix[0][2],rotmatrix[0][3]
    #### TODO DELETE -->> "old version" ---> mat[1][0:4] = rotmatrix[1][0],rotmatrix[1][1],rotmatrix[1][2],rotmatrix[1][3]
    #### TODO DELETE -->> "old version" ---> mat[2][0:4] = rotmatrix[2][0],rotmatrix[2][1],rotmatrix[2][2],rotmatrix[2][3]
    #### TODO DELETE -->> "old version" ---> mat[3][0:4] = rotmatrix[3][0],rotmatrix[3][1],rotmatrix[3][2],rotmatrix[3][3]
    #### TODO DELETE -->> "old version" ---> roll, pitch, yaw = mat.to_euler('ZYX')
    #### TODO DELETE -->> "old version" ---> x,y,z = mat.to_translation()

    # qui sotto corrisponde a
    # quat2eul([ 0.997785  -0.0381564  0.0358964  0.041007 ],'XYZ')
    # TODO se tutto funziona, si potrebbe provare di nuovo con mathutils
    roll = atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = asin(rotmatrix[0, 2])
    yaw = atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return np.array([x, y, z, roll, pitch, yaw])


def getRT(a, h, k):
    RT = np.array([[cos(a), sin(a), -h * cos(a) - k * sin(a)],
                   [-sin(a), cos(a), h * sin(a) - k * cos(a)],
                   [0., 0., 1.]])
    return RT


def rotate_point(p, center, angle):
    """Funzione per ruotare un punto
    :param p: Punto (x,y)
    :param center: Centro di rotazione (x,y)
    :param angle: angolo di rotazione
    :return: Punto routato (x,y)
    """
    s = sin(angle)
    c = cos(angle)
    x = p[0]
    y = p[1]
    center_x = center[0]
    center_y = center[1]
    x = x - center_x
    y = y - center_y
    new_x = x * c - y * s
    new_y = x * s + y * c
    new_x = new_x + center_x
    new_y = new_y + center_y
    return new_x, new_y


def bearing(latA, lonA, latB, lonB):
    """
    :param latA:
    :param lonA:
    :param latB:
    :param lonB:
    :return:
    """
    bearing = atan2(sin(lonB - lonA) * cos(latB), cos(latA) * sin(latB) - sin(latA) * cos(latB) * cos(lonB - lonA))
    bearing = bearing + 2.0 * pi
    while bearing > 2. * pi:
        bearing -= 2. * pi
    return bearing


def radians(deg):
    return (deg * pi) / 180.0


def degrees(rad):
    return rad * (180.0 / pi)


def teacher_network_pass(args, sample, model, criterion):
    if args.triplet:
        # Obtain sample values
        anchor = sample['anchor']  # OSM Type X
        positive = sample['positive']  # OSM Type X
        negative = sample['negative']  # OSM Type Y

        # Sent to the graphic card if posible
        if torch.cuda.is_available() and args.use_gpu:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        # Obtain predicion results
        out_anchor = model(anchor)
        out_positive = model(positive)
        out_negative = model(negative)

        # Calculate the loss
        loss = criterion(out_anchor, out_positive, out_negative)

        # Calculate the accuracy
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        result = (cos_sim(out_anchor, out_positive) + 1.0) * 0.5
        acc = torch.sum(result).item()

    else:  # classification
        # Obtain sample values
        data = sample['anchor']
        label = sample['label_anchor']

        # Sent to the graphic card if posible
        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        # Obtain predicion results
        output = model(data)

        # Calculate the loss
        loss = criterion(output, label)

        # Calculate the accuracy
        predict = torch.argmax(output, 1)
        label = label.cpu().numpy()
        predict = predict.cpu().numpy()

        acc = accuracy_score(label, predict)

    if args.triplet:
        return acc, loss
    else:
        return acc, loss, label, predict


def student_network_pass(args, sample, criterion, model, gtmodel=None):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    if args.triplet:
        if args.dataloader == 'triplet_OBB':
            anchor = sample['OSM_anchor']  # OSM Anchor
            positive = sample['BEV_positive']  # BEV Positive
            negative = sample['BEV_negative']  # BEV Negative
        if args.dataloader == 'triplet_BOO':
            anchor = sample['BEV_anchor']  # BEV Image
            positive = sample['OSM_positive']  # OSM Positive
            negative = sample['OSM_negative']  # OSM Negative
        if torch.cuda.is_available() and args.use_gpu:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        out_anchor = model(anchor)
        out_positive = model(positive)
        out_negative = model(negative)

        loss = criterion(out_anchor, out_positive, out_negative)

        result = ((cos_sim(out_anchor.squeeze(), out_positive.squeeze()) + 1.0) * 0.5)
        acc = torch.sum(result).item()

    elif args.embedding:
        data = sample['data']
        osm = sample['generated_osm']

        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            osm = osm.cuda()

        output = model(data)
        output_gt = gtmodel(osm)  # (Batch x 512) Tensor

        mask = torch.ones((64, 512)).cuda()
        loss = criterion(output, output_gt, mask)

        result = ((cos_sim(output.squeeze(), output_gt.squeeze()) + 1.0) * 0.5)
        acc = torch.sum(result).item()

    else:
        data = sample['data']
        label = sample['label']
        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        output = model(data)

        loss = criterion(output, label)

        predict = torch.argmax(output, 1)
        label = label.cpu().numpy()
        predict = predict.cpu().numpy()

        acc = accuracy_score(label, predict)

    if args.triplet or args.embedding:
        return acc, loss
    else:
        return acc, loss, label, predict


def init_function(worker_id, seed, epoch):
    """
    This method was copied from old Daniele's works, with some enhancements into the "main" part. This will be the code
    called from the dataloader. To initialize the seeds differently in every epoch, we use the following parameters

    Args:
        worker_id: will automagically appears from pytorch
        seed: this value is something you choose. usually we set to zero for reproducibility
        epoch: allow us to have a different seed every epoch

    Returns:

    """
    seed = seed.value + worker_id + epoch.value * 100
    # if you want to debug... print(f"\nInit worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
