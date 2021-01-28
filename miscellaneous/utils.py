import datetime
import json
import linecache
import os
import pickle
import random
import sys
from functools import reduce
from io import BytesIO
from math import asin, atan2, cos, pi, sin

from sklearn import svm

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.covariance import MinCovDet
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


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

    Requires image within 0..1 range

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


def teacher_network_pass(args, sample, model, criterion, gt_list=None):
    if args.triplet:
        # Obtain sample values
        anchor = sample['anchor']  # OSM Type X
        positive = sample['positive']  # OSM Type X
        negative = sample['negative']  # OSM Type Y
        label = sample['label_anchor']

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
        if gt_list is not None:
            predict = gt_triplet_validation(out_anchor, model, gt_list)
            acc = accuracy_score(label.squeeze().numpy(), predict)
        else:
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

    if args.triplet and gt_list is None:
        return acc, loss
    else:
        return acc, loss, label, predict


def student_network_pass(args, sample, criterion, model, gt_list=None, weights_param=None, return_embedding=False,
                         miner=None, acc_metric=None):
    embedding = None
    label = None
    predict = None

    if args.embedding:
        data = sample['data']
        label = sample['label']

        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()

        output = model(data)
        output_gt = gt_list[label.squeeze()]  # --> Embeddings centroid of the label

        # save the embedding vector to return it - used in testing
        if return_embedding:
            embedding = np.asarray(output.squeeze().cpu().detach().numpy())

        if args.lossfunction == 'triplet':
            neg_label = sample['neg_label']
            neg_output_gt = gt_list[neg_label.squeeze()]
            loss = criterion(output.squeeze(), output_gt.cuda(), neg_output_gt.cuda())  # --> 128 x 512
        else:
            loss = criterion(output.squeeze(), output_gt.cuda())  # --> 128 x 512

        if args.weighted:
            weights = torch.FloatTensor(weights_param)
            weighted_tensor = weights[label.squeeze()]
            loss = loss * weighted_tensor.cuda().unsqueeze(1)
            loss = loss.mean()

        if args.lossfunction == 'triplet':
            predict = gt_validation(output, gt_list)
        else:
            predict = gt_validation(output, gt_list, criterion)
        acc = accuracy_score(label.squeeze().numpy(), predict)

    elif args.triplet:
        anchor = sample['anchor']
        positive = sample['positive']
        negative = sample['negative']

        if torch.cuda.is_available() and args.use_gpu:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        out_anchor = model(anchor)
        out_positive = model(positive)
        out_negative = model(negative)

        loss = criterion(out_anchor, out_positive, out_negative)
        acc = acc_triplet_score(args, out_anchor, out_positive, out_negative)

    elif args.metric:
        data = sample['data']  # RGB images
        label = sample['label']
        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        embeddings = model(data)
        if args.miner:
            hard_pairs = miner(embeddings.squeeze(), label)  # El miner no funciona y no da ningun error
            loss = criterion(embeddings.squeeze(), label, hard_pairs)
        else:
            loss = criterion(embeddings.squeeze(), label)

        # acc is not a value is a dict
        acc_dict = acc_metric.get_accuracy(embeddings, embeddings, label, label, labelembeddings_come_from_same_source=True)
        # acc = acc_dict['r_precision']
        acc = acc_dict['mean_average_precision_at_r']

    else:
        data = sample['data']
        label = sample['label']
        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        output = model(data)

        # save the embedding vector to return it - used in testing
        if return_embedding:
            embedding = np.asarray(output.squeeze().cpu().detach().numpy())

        loss = criterion(output, label)

        predict = torch.argmax(output, 1)
        label = label.cpu().numpy()
        predict = predict.cpu().numpy()

        acc = accuracy_score(label, predict)

    return acc, loss, label, predict, embedding


def lstm_network_pass(args, sample, criterion, model, lstm):
    feature_list = []
    seq_list = []
    len_list = []
    batch = sample['sequence']  # --> Batch x Seq x W x H
    label = sample['label']  # --> Batch x 1

    with torch.no_grad():
        for sequence in batch:
            for img in sequence:
                features = model(img)  # --> (1x512)
                feature_list.append(features)
            seq_tensor = torch.stack(feature_list, dim=1)  # --> (seq_len x 512)
            seq_list.append(seq_tensor)
            len_list.append(len(sequence))
        padded_batch = pad_sequence(seq_list, batch_first=True).size()
        packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                                   batch_first=True)  # --> (Batch x Max_seq_len x 512)

    result = lstm(packed_padded_batch)

    loss = criterion(result, label)

    predict = torch.argmax(result, 1)
    label = label.numpy()
    predict = predict.cpu().numpy()

    acc = accuracy_score(label, predict)

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

    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def svm_generator(args, model, dataloader_train, dataloader_val):
    svm_path = args.load_path.replace('.pth', 'svm.sav')
    if os.path.isfile(svm_path):
        print('SVM already trained in => {}'.format(svm_path))
        classifier = pickle.load(open(svm_path, 'rb'))
    else:
        print('training SVM classifier\n')
        print('svm model will be saved in : {}\n'.format(svm_path))
        features, labels = embb_data(args, model, dataloader_train, dataloader_val)
        classifier = svm_train(features, labels, mode=args.svm_mode)
        pickle.dump(classifier, open(svm_path, 'wb'))

    return classifier


def svm_train(features, labels, mode='Linear'):
    assert features.shape[0] == labels.shape[
        0], 'The number of feature vectors {} should be same as number of labels  {}'.format(
        features.shape[0], labels.shape[0])
    if mode == 'Linear':
        classifier = svm.LinearSVC(class_weight='balanced')
        classifier.fit(features, labels)
    else:
        classifier = svm.SVC(decision_function_shape='ovo', class_weight='balanced')
        classifier.fit(features, labels)
        classifier.decision_function_shape = "ovr"

    return classifier


def embb_data(args, model, dataloader_train, dataloader_val, save=False):
    embeddingRecord = np.empty((0, 512), dtype=np.float32)
    labelRecord = np.array([], dtype=np.uint8)
    model.eval()
    with torch.no_grad():

        for sample in dataloader_train:
            if args.embedding:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data)  # --> (Batch x 512)
            embeddingRecord = np.append(embeddingRecord, output.squeeze().cpu().numpy(), axis=0)
            labelRecord = np.append(labelRecord, label.squeeze())

        for sample in dataloader_val:
            if args.embedding:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data)  # --> (Batch x 512)

            embeddingRecord = np.append(embeddingRecord, output.squeeze().cpu().numpy(), axis=0)
            labelRecord = np.append(labelRecord, label)

    if save:
        print('Saving embeddings')
        with open('./trainedmodels/embeddings/embeddings.npy', 'wb') as f:
            np.save(f, embeddingRecord)
        with open('./trainedmodels/embeddings/labels.npy', 'wb') as f:
            np.save(f, labelRecord)

    return embeddingRecord, labelRecord


def svm_testing(args, model, dataloader_test, classifier):
    print('Start svm testing')

    label_list = []
    prediction_list = []

    with torch.no_grad():
        model.eval()

        for sample in dataloader_test:
            if args.embedding:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data)  # --> (1 x 512)
            dec = classifier.decision_function(output)
            prediction = dec.shape[1]

            label_list.append(label)
            prediction_list.append(prediction)

        conf_matrix = pd.crosstab(np.array(label_list), np.array(prediction_list), rownames=['Actual'],
                                  colnames=['Predicted'],
                                  normalize='index')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
        acc = accuracy_score(np.array(label_list), np.array(prediction_list))
        print('Accuracy for test : %f\n' % acc)
        return conf_matrix, acc


def covmatrix_generator(args, model, dataloader_train, dataloader_val):
    features, labels = embb_data(args, model, dataloader_train, dataloader_val)
    clusters = {}
    covariances = {}
    for lbl in np.unique(labels):
        clusters[lbl] = features[labels == lbl, :]
        covariances[lbl] = MinCovDet(random_state=0).fit(clusters[lbl])

    return covariances


def mahalanovis_testing(args, model, dataloader_test, covariances):
    print('Start mahalanobis testing')

    label_list = []
    prediction_list = []
    distance_list = []

    with torch.no_grad():
        model.eval()

        for sample in dataloader_test:
            if args.embedding:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data)  # --> (1 x 512)
            for lbl in range(7):
                dist = covariances[lbl].mahalanobis(output)
                distance_list.append(dist.item())
            prediction = np.argmin(np.array(distance_list))

            label_list.append(label)
            prediction_list.append(prediction)

        conf_matrix = pd.crosstab(np.array(label_list), np.array(prediction_list), rownames=['Actual'],
                                  colnames=['Predicted'],
                                  normalize='index')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
        acc = accuracy_score(np.array(label_list), np.array(prediction_list))
        print('Accuracy for test : %f\n' % acc)
        return conf_matrix, acc


def gt_validation(output, gt_list, criterion=None):
    l = []
    if criterion is None:
        criterion = nn.MSELoss()
    for batch_item in output:
        for gt in gt_list:
            l.append(criterion(batch_item.squeeze(), gt.cuda()).mean().item())
    nplist = np.array(l)
    nplist = nplist.reshape(-1, 7)
    classification = np.argmin(nplist, axis=1)

    return classification


def gt_triplet_validation(out_anchor, model, gt_list):
    l = []
    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    for batch_item in out_anchor:
        for gt in gt_list:
            gt = gt.cuda()
            gt_prediction = model(gt)
            l.append(criterion(batch_item, gt_prediction).item())  # Revisar esto
    nplist = np.array(l)
    nplist = nplist.reshape(-1, 7)
    classification = np.argmin(nplist, axis=1)

    return classification


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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def acc_triplet_score(args, out_anchor, out_positive, out_negative):
    if args.distance_function == 'pairwise':
        distance_func = torch.nn.PairwiseDistance()
        distance_pos = distance_func(out_anchor, out_positive)  # (Nx512) · (Nx512) --> (N)
        distance_neg = distance_func(out_anchor, out_negative)
        acc = torch.sum(distance_pos < distance_neg) / args.batch_size
    elif args.distance_function == 'cosine':
        distance_func = torch.nn.CosineSimilarity()
        distance_pos = 1.0 - distance_func(out_anchor, out_positive)
        distance_neg = 1.0 - distance_func(out_anchor, out_negative)
        acc = torch.sum(distance_pos < distance_neg) / args.batch_size

    return acc
