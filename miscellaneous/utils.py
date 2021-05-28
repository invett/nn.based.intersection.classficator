import datetime
import time
import json
import linecache
import os
import pickle
import random
import sys
from functools import reduce
from io import BytesIO
from math import asin, atan2, cos, pi, sin

from pytorch_metric_learning import testers
from scipy.spatial import distance
from scipy.special import softmax
from sklearn import svm
import pathlib
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.covariance import MinCovDet
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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
    try:

        figdata = BytesIO()
        plt.savefig(figdata, format='png', dpi=600)
        URI = 'https://api.telegram.org/bot' + telegram_token + '/sendPhoto?chat_id=' + telegram_channel + "&caption=" + str(
            datetime.datetime.now()) + "\n" + description
        pic = {'photo': ("Foto", figdata.getvalue(), 'image/png')}
        response = requests.get(URI, files=pic)

        return json.loads(response.content)['ok']

    except:
        print('error sending telegram message')

        return -1


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
        if args.model == 'inception_v3' and model.training:
            out_anchor, aux_anchor = model(anchor)
            out_positive, aux_positive = model(positive)
            out_negative, aux_negative = model(negative)
        else:
            out_anchor = model(anchor)
            out_positive = model(positive)
            out_negative = model(negative)

        # Calculate the loss
        loss = criterion(out_anchor, out_positive, out_negative)
        if args.model == 'inception_v3' and model.training:
            loss_aux = criterion(aux_anchor, aux_positive, aux_negative)
            loss = loss + loss_aux * 0.4
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
        if args.model == 'inception_v3' and model.training:
            loss_aux = criterion(output[1], label)
            loss = criterion(output[0], label)
            loss = loss + loss_aux * 0.4
        else:
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
        if args.model == 'inception_v3' and model.training:
            output = output[0]
            output_aux = output[1]
        output_gt = gt_list[label.squeeze()]  # --> Embeddings centroid of the label

        # save the embedding vector to return it - used in testing
        if return_embedding:
            embedding = np.asarray(output.squeeze().cpu().detach().numpy())

        if args.lossfunction == 'triplet':
            neg_label = sample['neg_label']
            neg_output_gt = gt_list[neg_label.squeeze()]
            loss = criterion(output.squeeze(), output_gt.cuda(), neg_output_gt.cuda())  # --> 128 x 512
        else:
            if args.model == 'inception_v3' and model.training:
                loss_aux = criterion(output_aux.squeeze(), output_gt.cuda())
                loss = criterion(output.squeeze(), output_gt.cuda())
                loss = loss + loss_aux * 0.4
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
        if miner is not None:
            hard_pairs = miner(embeddings.squeeze(), label)
            loss = criterion(embeddings.squeeze(), label, hard_pairs)
        else:
            loss = criterion(embeddings.squeeze(), label)

        # acc is not a value is a dict
        acc = acc_metric.get_accuracy(embeddings.squeeze().detach().cpu().numpy(),
                                      embeddings.squeeze().detach().cpu().numpy(), label.cpu().numpy(),
                                      label.cpu().numpy(), embeddings_come_from_same_source=True)

        # TODO : no se puede hacer algo asi?
        # conf_matrix = pd.crosstab(np.array(label_list),
        #                           np.array(prediction_list),
        #                           rownames=['Actual'],
        #                           colnames=['Predicted'],
        #                           normalize='index')
        # conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)

    else:
        data = sample['data']
        label = sample['label']
        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        if args.get_scores:
            criterion = nn.NLLLoss()

        output = model(data)

        # save the embedding vector to return it - used in testing
        if return_embedding or args.get_scores:
            embedding = np.asarray(output.squeeze().cpu().detach().numpy())

        loss = criterion(output, label)

        predict = torch.argmax(output, 1)
        label = label.cpu().numpy()
        predict = predict.cpu().numpy()

        acc = accuracy_score(label, predict)

    return acc, loss, label, predict, embedding


def lstm_network_pass(args, batch, criterion, model, lstm, miner=None, acc_metric=None):
    seq_list = []
    len_list = []
    predict = None
    label = torch.tensor([int(sequence['label']) for sequence in batch]).cuda()  # Unpack label values

    with torch.no_grad():
        for sequence in batch:
            seq_tensor = model(torch.stack(sequence['sequence']).cuda())  # return--> (img_seq x 512)
            seq_list.append(seq_tensor.squeeze())
            len_list.append(len(sequence['sequence']))

    padded_batch = pad_sequence(seq_list, batch_first=True)
    packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                               batch_first=True,
                                               enforce_sorted=False)  # --> (Batch x Max_seq_len x 512)

    prediction, output = lstm(packed_padded_batch)

    if args.export_data:
        # Output contains a packed sequence with the prediction in each timestamp --> (seq_len x batch x hidden_size)
        # Prediction contains the prediction in the last timestamp --> (batch x hidden_size)

        # Unpack the sequence (PACK -> PAD...) and then
        #   output_overall          => BATCH x MAX_SEQ_LEN x LSTM_HIDDEN_SIZE(32) example 47x50x32
        #   len_of_each_sequence    => how many 'actual' values in each of the batch, since we padded with 0 (pad_sequence)
        #
        # so, with overall_output[0, 0:12, :] we retrieve:
        #       1. given the first sequence (0)
        #       2. retrieve all 'actual' elements ... 12 elements, which corresponds to len_of_each_sequence(0)
        #       3. and select all the hidden_vector (hidden_size) of the LSTM (32 for example).
        #
        # these are the 'vector' that the LSTM uses to evaluate the actual prediction by means of the FC network!

        output_overall, len_of_each_sequence = pad_packed_sequence(output, batch_first=True)

        all_predictions = lstm.export_predictions(output_overall, len_of_each_sequence)

        # create list to export
        flat_all_filenames = [filename for item in batch for filename in item['path_of_original_images']]
        flat_all_labels = [item for sublist in
                           [np.ones(len(item['sequence']), int) * int(item['label']) for item in batch]
                           for item in sublist]
        flat_all_predictions = [item for sublist in all_predictions for item in sublist]
        export_data = [flat_all_filenames, flat_all_labels, flat_all_predictions]

        # export_data: this list contains data to create a file that is similar to 'test_list.txt' used
        # in txt_dataloader. will be used to compare RESNET vs LSTM as they are already in 'per-sequence' format.

        filename = '/tmp/' + str(int(time.time())) + '_' + os.path.splitext(os.path.split(args.dataset_test)[1])[
            0] + '_lstm_export_svm' + \
                   os.path.splitext(os.path.split(args.dataset_test)[1])[1]
        print('\nsaving data in: ' + filename)

        # create filename
        with open(filename, "w") as output:
            for i in range(len(export_data[0])):
                line = export_data[0][i] + ';' + str(export_data[1][i]) + ';' + str(export_data[2][i]) + '\n'
                output.write(line)

    if args.metric:
        if miner is not None:
            hard_pairs = miner(prediction, label)
            loss = criterion(prediction, label, hard_pairs)
        else:
            loss = criterion(prediction, label)

        # acc is not a value is a dict
        acc = acc_metric.get_accuracy(prediction.detach().cpu().numpy(),
                                      prediction.detach().cpu().numpy(), label.cpu().numpy(),
                                      label.cpu().numpy(), embeddings_come_from_same_source=True)
    else:
        loss = criterion(prediction, label)

        predict = torch.argmax(prediction, 1)
        label = label.cpu().numpy()
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


def svm_generator(args, model, dataloader_train=None, dataloader_val=None, LSTM=None):
    if args.svm_mode == 'Linear':
        svm_path = args.load_path.replace('.pth', '.lsvm.sav')
    elif args.svm_mode == 'ovo':
        svm_path = args.load_path.replace('.pth', '.osvm.sav')
    else:
        print("The SVM mode is not implemented: " + args.svm_mode)
        exit(1)

    if os.path.isfile(svm_path):
        print('SVM already trained in => {}'.format(svm_path))
        classifier = pickle.load(open(svm_path, 'rb'))
    else:
        print('training SVM classifier\n')
        print('svm model will be saved in : {}\n'.format(svm_path))
        if not args.metric and LSTM is None:
            features, labels = embb_data(args, model, dataloader_train, dataloader_val)
        else:
            if LSTM is not None:
                features, labels = embb_data_lstm(model, dataloader_train, dataloader_val, LSTM=LSTM)
            else:
                train_embeddings, train_labels = get_all_embeddings(dataloader_train, model)
                val_embeddings, val_labels = get_all_embeddings(dataloader_val, model)
                features = np.vstack((train_embeddings, val_embeddings))
                labels = np.vstack((train_labels, val_labels))

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


def embb_data_lstm(model, dataloader_train, dataloader_val, LSTM=None):
    embeddingRecord = []
    labelRecord = []

    LSTM.eval()

    with torch.no_grad():
        for batch in dataloader_train:
            seq_list = []
            len_list = []
            label_list = []
            for sequence in batch:
                seq_tensor = model(torch.stack(sequence['sequence']).cuda())
                seq_list.append(seq_tensor.squeeze())
                len_list.append(len(sequence['sequence']))
                label_list.append(int(sequence['label']))

            padded_batch = pad_sequence(seq_list, batch_first=True)
            packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                                       batch_first=True,
                                                       enforce_sorted=False)  # --> (Batch x Max_seq_len x 512)

            prediction, output = LSTM(packed_padded_batch)
            # Output contains a packed sequence with the prediction in each timestamp --> (seq_len x batch x hidden_size)
            # Prediction contains the prediction in the last timestamp --> (batch x hidden_size)

            # output, lens_output = pad_packed_sequence(output, batch_first=True)  ## que hacer con output??

            embeddingRecord.append(prediction.cpu().numpy())
            labelRecord.append(np.expand_dims(np.array(label_list), axis=1))

        for batch in dataloader_val:
            seq_list = []
            len_list = []
            label_list = []
            for sequence in batch:
                seq_tensor = model(torch.stack(sequence['sequence']).cuda())
                seq_list.append(seq_tensor.squeeze())
                len_list.append(len(sequence['sequence']))
                label_list.append(int(sequence['label']))

            padded_batch = pad_sequence(seq_list, batch_first=True)
            packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                                       batch_first=True,
                                                       enforce_sorted=False)  # --> (Batch x Max_seq_len x 512)

            prediction, output = LSTM(packed_padded_batch)
            # Output contains a packed sequence with the prediction in each timestamp --> (seq_len x batch x hidden_size)
            # Prediction contains the prediction in the last timestamp --> (batch x hidden_size)

            # output, lens_output = pad_packed_sequence(output, batch_first=True)  ## que hacer con output??

            embeddingRecord.append(prediction.cpu().numpy())
            labelRecord.append(np.expand_dims(np.array(label_list), axis=1))

    return np.vstack(embeddingRecord), np.vstack(labelRecord)


def svm_testing(args, model, dataloader_test, classifier, probs=False):
    print('Start svm testing')

    # defining the lists that will be used to export data, for RESNET vs LSTM comparison
    export_filenames = []
    export_gt_labels = []
    export_prediction_list = []

    label_list = []
    prediction_list = []
    score_list = []

    with torch.no_grad():
        model.eval()

        # all_output = [] TODO: seems that we don't use this...

        for sample in dataloader_test:
            if args.embedding or args.metric:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data)  # --> (1 x 512)
            # all_output.append(output.cpu().numpy()) TODO: seems that we don't use this...
            dec = classifier.decision_function(output.cpu().numpy())
            prediction = np.argmax(dec, axis=1)

            export_filenames.extend(sample['path_of_original_image'])
            label_list.append(label.cpu().numpy())
            prediction_list.append(prediction)
            score_list.append(dec)

        conf_matrix = pd.crosstab(np.hstack(label_list), np.hstack(prediction_list), rownames=['Actual'],
                                  colnames=['Predicted'],
                                  normalize='index')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
        acc = accuracy_score(np.hstack(label_list), np.hstack(prediction_list))
        print('Accuracy for test : %f\n' % acc)

        # Score for testing dataset
        logit_score = np.vstack(score_list)
        prob_score = softmax(logit_score, axis=1)

        # these three lists will be used to create a file similar to the 'test_list.txt' used with the txt_dataloader.
        # these will be used to evaluate RESNET vs LSTM on a 'per-sequence' basis
        export_filenames = export_filenames
        [export_gt_labels.extend(i) for i in label_list]
        [export_prediction_list.extend(i) for i in prediction_list]
        export_overall = [export_filenames, export_gt_labels, export_prediction_list]

        if probs:
            return conf_matrix, acc, export_overall, (prob_score, logit_score)
        else:
            return conf_matrix, acc, export_overall


def svm_testing_lstm(model, dataloader_test, classifier, LSTM):
    prediction_list = []
    label_list = []

    LSTM.eval()

    with torch.no_grad():
        for batch in dataloader_test:
            seq_list = []
            len_list = []
            seq_label_list = []
            for sequence in batch:
                seq_tensor = model(torch.stack(sequence['sequence']).cuda())
                seq_list.append(seq_tensor.squeeze())
                len_list.append(len(sequence['sequence']))
                seq_label_list.append(int(sequence['label']))

            padded_batch = pad_sequence(seq_list, batch_first=True)
            packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                                       batch_first=True,
                                                       enforce_sorted=False)  # --> (Batch x Max_seq_len x 512)

            prediction, output = LSTM(packed_padded_batch)
            # Output contains a packed sequence with the prediction in each timestamp --> (seq_len x batch x hidden_size)
            # Prediction contains the prediction in the last timestamp --> (batch x hidden_size)

            # output, lens_output = pad_packed_sequence(output, batch_first=True)

            dec = classifier.decision_function(prediction.cpu().numpy())
            prediction = np.argmax(dec, axis=1)

            prediction_list.append(prediction)
            label_list.append(np.array(seq_label_list))

    conf_matrix = pd.crosstab(np.hstack(label_list), np.hstack(prediction_list), rownames=['Actual'],
                              colnames=['Predicted'],
                              normalize='index')
    conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
    acc = accuracy_score(np.hstack(label_list), np.hstack(prediction_list))
    print('Accuracy for test : %f\n' % acc)
    return conf_matrix, acc


def covmatrix_generator(args, model, dataloader_train=None, dataloader_val=None, LSTM=None):
    cov_path = args.load_path.replace('.pth', '.cov.sav')
    if os.path.isfile(cov_path):
        print('Covariance matriz already saved in => {}'.format(cov_path))
        covariances = pickle.load(open(cov_path, 'rb'))
    else:
        if not args.metric:
            features, labels = embb_data(args, model, dataloader_train, dataloader_val)
        else:
            if LSTM is not None:
                features, labels = embb_data_lstm(model, dataloader_train, dataloader_val, LSTM=LSTM)
            else:
                train_embeddings, train_labels = get_all_embeddings(dataloader_train, model)
                val_embeddings, val_labels = get_all_embeddings(dataloader_val, model)
                features = np.vstack((train_embeddings, val_embeddings))
                labels = np.vstack((train_labels, val_labels))

        clusters = {}
        covariances = {}
        print(datetime.datetime.now())
        for lbl in np.unique(labels):
            print(datetime.datetime.now())
            clusters[lbl] = features[(labels == lbl).squeeze(), :]
            covariances[lbl] = MinCovDet(random_state=0).fit(clusters[lbl])
        print(datetime.datetime.now())

        pickle.dump(covariances, open(cov_path, 'wb'))

    return covariances


def mahalanobis_testing(args, model, dataloader_test, covariances):
    print('Start mahalanobis testing')

    # defining the lists that will be used to export data, for RESNET vs LSTM comparison
    export_filenames = []
    export_gt_labels = []
    export_prediction_list = []

    label_list = []
    prediction_list = []

    with torch.no_grad():
        model.eval()
        for sample in dataloader_test:
            if args.embedding or args.metric:
                data = sample['data']
                label = sample['label']

            if args.triplet:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()

            output = model(data).cpu().numpy()  # --> (BATCH x 512)
            distance_list = []
            for lbl in range(7):
                dist = covariances[lbl].mahalanobis(output)
                # distance_list.append(dist.item())
                distance_list.append(dist)
            # prediction = np.argmin(np.array(distance_list))
            prediction = np.argmin(distance_list, axis=0)  # i want 64 labels, given from 7x64

            export_filenames.extend(sample['path_of_original_image'])
            label_list.append(label.cpu().numpy())
            prediction_list.append(prediction)

        conf_matrix = pd.crosstab(np.hstack(label_list), np.hstack(prediction_list), rownames=['Actual'],
                                  colnames=['Predicted'],
                                  normalize='index')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
        acc = accuracy_score(np.hstack(label_list), np.hstack(prediction_list))
        print('Accuracy for test : %f\n' % acc)

        # these three lists will be used to create a file similar to the 'test_list.txt' used with the txt_dataloader.
        # these will be used to evaluate RESNET vs LSTM on a 'per-sequence' basis
        export_filenames = export_filenames
        [export_gt_labels.extend(i) for i in label_list]
        [export_prediction_list.extend(i) for i in prediction_list]
        export_overall = [export_filenames, export_gt_labels, export_prediction_list]

        return conf_matrix, acc, export_overall


def mahalanobis_testing_lstm(model, dataloader_test, covariances, LSTM=None):
    print('Start mahalanobis testing LSTM')

    prediction_list = []
    label_list = []

    LSTM.eval()

    with torch.no_grad():
        for batch in dataloader_test:
            seq_list = []
            len_list = []
            seq_label_list = []
            for sequence in batch:
                seq_tensor = model(torch.stack(sequence['sequence']).cuda())
                seq_list.append(seq_tensor.squeeze())
                len_list.append(len(sequence['sequence']))
                seq_label_list.append(int(sequence['label']))

            padded_batch = pad_sequence(seq_list, batch_first=True)
            packed_padded_batch = pack_padded_sequence(padded_batch, len_list,
                                                       batch_first=True,
                                                       enforce_sorted=False)  # --> (Batch x Max_seq_len x 512)

            prediction, output = LSTM(packed_padded_batch)
            # Output contains a packed sequence with the prediction in each timestamp --> (seq_len x batch x hidden_size)
            # Prediction contains the prediction in the last timestamp --> (batch x hidden_size)

            # output, lens_output = pad_packed_sequence(output, batch_first=True)

            distance_list = []
            for lbl in range(7):
                dist = covariances[lbl].mahalanobis(prediction.cpu().numpy())
                distance_list.append(dist)
            prediction = np.argmin(distance_list, axis=0)  # i want 64 labels, given from 7x64

            prediction_list.append(prediction)
            label_list.append(np.array(seq_label_list))

        conf_matrix = pd.crosstab(np.hstack(label_list), np.hstack(prediction_list), rownames=['Actual'],
                                  colnames=['Predicted'],
                                  normalize='index')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
        acc = accuracy_score(np.hstack(label_list), np.hstack(prediction_list))
        print('Accuracy for test : %f\n' % acc)

        return conf_matrix, acc


def get_all_embeddings(dataloader, model):
    """
    https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#basetester

    ### convenient function from pytorch-metric-learning ###
    def get_all_embeddings(dataset, model):
        tester = testers.BaseTester()
        return tester.get_all_embeddings(dataset, model)

    :param dataloader: the dataloader to use
    :param model: the model to use
    :return:
    """

    tester = testers.BaseTester(normalize_embeddings=False, data_and_label_getter=image_getter)
    return tester.get_all_embeddings(dataloader.dataset, model)


def image_getter(sample):
    """

    A function that takes the output of your dataset's __getitem__ function, and returns a tuple of (data, labels).
    If None, then it is assumed that __getitem__ returns (data, labels).

    :param sample: will be the output of the getitem from our dataloader
    :return: tuple (data, labels)
    """

    result = (sample['data'], sample['label'])
    return result


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
    model.eval()
    with torch.no_grad():
        criterion = torch.nn.SmoothL1Loss(reduction='mean')
        for batch_item in out_anchor:
            for gt in gt_list:
                gt = gt.cuda()
                gt_prediction = model(gt)
                if isinstance(gt_prediction, tuple):
                    gt_prediction = gt_prediction[0]
                l.append(criterion(batch_item, gt_prediction).item())
        nplist = np.array(l)
        nplist = nplist.reshape(-1, 7)
        classification = np.argmin(nplist, axis=1)
    model.train()
    return classification


def get_distances(dataloader_test, model, centroid_list):
    distances = []
    model.eval()
    with torch.no_grad():
        distance = nn.PairwiseDistance(p=2)
        for sample in dataloader_test:
            output = model(sample)
            for batch_item in output:
                for ct in centroid_list:
                    distances.append(distance(batch_item.squeeze(), ct.cuda()))
    npdist = np.array(distances)
    npdist = npdist.reshape(-1, 7)

    return npdist


def get_distances_embb(embbedings, centroid_list):
    distances = []
    for embedding in embbedings:
        for ct in centroid_list:
            distances.append(distance.euclidean(embedding, ct))

    npdist = np.array(distances)
    npdist = npdist.reshape(-1, 7)

    return npdist

def get_distances_embb_torch(embbedings, centroid_list):

    return torch.cdist(embbedings, centroid_list)


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


def split_dataset(annotations, files, prefix_filename='prefix_', save_folder='/tmp',
                  overwrite_i_dont_care=False, extract_field_from_path=-1, threshold=5):
    '''

    This function is called from the labelling-script and similar scripts.

    Originally called "summary" inside the labelling script, was improved in split.kitti360.warpings-2.py in order to
    work with SEQUENCES. Then since it was used in different scripts we moved the routine here and changed its name.

    Original documentation:

    This script takes the some list generated with the labelling script and divided it in three parts to be used in *BOTH*
    RESNET and LSTM.

    Initially created to divide KITTI360 in three splits, it can be used to split other datasets as well.

      NEEDED: one txt in the follwing form:

      (base) ballardini@ballardini-T14:~/Desktop/alcala-12.02.2021.000$ head all_frames_labelled_focus_and_c4.txt
      122302AA/0000000791.png;0
      122302AA/0000000792.png;0
      122302AA/0000000793.png;0
      122302AA/0000000794.png;0
      122302AA/0000000795.png;0
      122302AA/0000000796.png;0
      122302AA/0000000797.png;0
      122302AA/0000000798.png;0
      122302AA/0000000799.png;0
      122302AA/0000000800.png;0

      KITTI360 WARNING! if filename contains '_' then use this line

      current_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[7])

      and create a three files:
                                  prefix_train_list.txt
                                  prefix_validation_list.txt
                                  prefix_test_list.txt

    Args:
        annotations:
        files:
        prefix_filename:
        save_folder:
        overwrite_i_dont_care:
        extract_field_from_path:
        threshold: min length of the "sequence" . put 0 to use all frames

    Returns: nothing

    '''
    debug_this_funcion = False

    print("Computing annotations...")
    type_0 = sum(sum(i == 0 for i in j) for j in annotations)
    type_1 = sum(sum(i == 1 for i in j) for j in annotations)
    type_2 = sum(sum(i == 2 for i in j) for j in annotations)
    type_3 = sum(sum(i == 3 for i in j) for j in annotations)
    type_4 = sum(sum(i == 4 for i in j) for j in annotations)
    type_5 = sum(sum(i == 5 for i in j) for j in annotations)
    type_6 = sum(sum(i == 6 for i in j) for j in annotations)
    type_x_frames = type_0 + type_1 + type_2 + type_3 + type_4 + type_5 + type_6

    type_x_sequences = [[], [], [], [], [], [], []]
    type_all_sequences = []

    sequences = 0
    sequences_frame_number = []
    file_added = 0
    for i in range(len(annotations)):
        current_sequence_frames = 0
        prev_frame_class = None
        prev_frame_number = None

        current_sequence_filenames = []

        # iterate both annotations and filename lists together, contains labels and frame-names
        for frame_class, frame_filename in zip(annotations[i], files[i]):

            if pathlib.Path(frame_filename).suffix != '.png':
                print('WARNING!!! Hey take care! Folders with PNGs should contain PNG only! --> ' + frame_filename)
                continue

            if debug_this_funcion:
                print(frame_filename)

            if current_sequence_frames == 0 and frame_class == -1:
                continue

            current_frame_number = getFrameNumber(extract_field_from_path, frame_filename)

            if prev_frame_number is None:
                prev_frame_number = current_frame_number - 1

            # check for sequence. if the current frame number is not the previous+1, then we have a new sequence.
            # we need to enter here also if we're analyzing the last frame, or we will lost ALL the frames in the last
            # sequence of the folder
            if not (prev_frame_class is None or (frame_class == prev_frame_class)) or (frame_filename == files[i][-1]) \
                    or (current_frame_number != prev_frame_number + 1):

                # if we're in the last frame, we need to re-check if the last frame belongs to the sequence again
                if frame_filename == files[i][-1]:
                    if frame_class == prev_frame_class:
                        current_sequence_frames = current_sequence_frames + 1
                        current_sequence_filenames.append(frame_filename)
                else:
                    if (frame_class == prev_frame_class) and (current_frame_number == prev_frame_number + 1):
                        current_sequence_filenames.append(frame_filename)

                if prev_frame_class == 0:
                    type_x_sequences[0].append(current_sequence_filenames.copy())
                if prev_frame_class == 1:
                    type_x_sequences[1].append(current_sequence_filenames.copy())
                if prev_frame_class == 2:
                    type_x_sequences[2].append(current_sequence_filenames.copy())
                if prev_frame_class == 3:
                    type_x_sequences[3].append(current_sequence_filenames.copy())
                if prev_frame_class == 4:
                    type_x_sequences[4].append(current_sequence_filenames.copy())
                if prev_frame_class == 5:
                    type_x_sequences[5].append(current_sequence_filenames.copy())
                if prev_frame_class == 6:
                    type_x_sequences[6].append(current_sequence_filenames.copy())

                if prev_frame_class != None:
                    type_all_sequences.append([current_sequence_filenames.copy(), prev_frame_class])

                if len(current_sequence_filenames) != current_sequence_frames:
                    print("Error")
                sequences_frame_number.append(current_sequence_frames)
                prev_frame_class = None
                prev_frame_number = None
                current_sequence_frames = 0
                sequences += 1
                current_sequence_filenames = []
                continue

            file_added = file_added + 1
            current_sequence_frames = current_sequence_frames + 1
            prev_frame_class = frame_class
            prev_frame_number = getFrameNumber(extract_field_from_path, frame_filename)
            current_sequence_filenames.append(frame_filename)

    # some sequences of kitti 360 have only one frame, others 2 or 3 ... we need to prune these sequences
    # set as parameter ---> threshold = 5
    excluded_counter = 0
    type_x_sequences_ = [[], [], [], [], [], [], []]
    for index, val in enumerate(type_x_sequences):
        for sq in val:
            if len(sq) > threshold:
                type_x_sequences_[index].append(sq)
            else:
                print("Skipping sequence with #frames: " + str(len(sq)))
                print("Frames skipped are: ")
                print(sq)
                excluded_counter = excluded_counter + len(sq)
    print("Total frames skipped: " + str(excluded_counter))

    type_x_sequences = type_x_sequences_.copy()

    print("Type0 seq/frames:", len(type_x_sequences[0]), "/", type_0, "\t-\t", [len(i) for i in type_x_sequences[0]])
    print("Type1 seq/frames:", len(type_x_sequences[1]), "/", type_1, "\t-\t", [len(i) for i in type_x_sequences[1]])
    print("Type2 seq/frames:", len(type_x_sequences[2]), "/", type_2, "\t-\t", [len(i) for i in type_x_sequences[2]])
    print("Type3 seq/frames:", len(type_x_sequences[3]), "/", type_3, "\t-\t", [len(i) for i in type_x_sequences[3]])
    print("Type4 seq/frames:", len(type_x_sequences[4]), "/", type_4, "\t-\t", [len(i) for i in type_x_sequences[4]])
    print("Type5 seq/frames:", len(type_x_sequences[5]), "/", type_5, "\t-\t", [len(i) for i in type_x_sequences[5]])
    print("Type6 seq/frames:", len(type_x_sequences[6]), "/", type_6, "\t-\t", [len(i) for i in type_x_sequences[6]])
    print("Overall Sequences: ", sequences)
    print("The number of frames associated to each sequence is: ", sequences_frame_number)
    print("Sum of frames in seq ", sum(sequences_frame_number), '/', type_x_frames)

    train_list = []
    validation_list = []
    test_list = []

    for i in range(7):
        # split the sequences, not the files!

        tosplit = type_x_sequences[i].copy()
        random.shuffle(tosplit)

        # This creates a ragged nested sequence warning ... tried to resolve, but didn't work
        split_train_val_test = np.array(np.split(tosplit, [int(len(tosplit) * 0.7), int(len(tosplit) * 0.9)]),
                                        dtype="object").tolist()

        # and append those lists for each of the train/val/test sets ; don't forget to add also the label at the end
        # so the file will have namefile;label

        base_folder = ''

        for split_train in split_train_val_test[0]:
            for filename in split_train:
                if not os.path.isabs(filename):
                    towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                else:
                    towrite = filename
                train_list.append(towrite + ';' + str(i))
        for split_valid in split_train_val_test[1]:
            for filename in split_valid:
                if not os.path.isabs(filename):
                    towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                else:
                    towrite = filename
                validation_list.append(towrite + ';' + str(i))
        for split_test in split_train_val_test[2]:
            for filename in split_test:
                if not os.path.isabs(filename):
                    towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                else:
                    towrite = filename
                test_list.append(towrite + ';' + str(i))

    print("Frames for Train/Val/Test: ", len(train_list), "/", len(validation_list), "/", len(test_list), "\tTot: ",
          len(train_list) + len(validation_list) + len(test_list), ' -- Skipping ', str(excluded_counter), 'frame(s): ',
          len(train_list) + len(validation_list) + len(test_list) - excluded_counter)

    # save the lists using the save_folder as root
    train_filename = prefix_filename + 'train_list.txt'
    validation_filename = prefix_filename + 'validation_list.txt'
    test_filename = prefix_filename + 'test_list.txt'
    all_filename = prefix_filename + 'all_list.txt'

    if not overwrite_i_dont_care:
        if os.path.isfile(os.path.join(save_folder, train_filename)):
            print('File already exists: ' + os.path.join(save_folder, train_filename))
            exit(1)
        if os.path.isfile(os.path.join(save_folder, validation_filename)):
            print('File already exists' + os.path.join(save_folder, validation_filename))
            exit(1)
        if os.path.isfile(os.path.join(save_folder, test_filename)):
            print('File already exists' + os.path.join(save_folder, test_filename))
            exit(1)
        if os.path.isfile(os.path.join(save_folder, all_filename)):
            print('File already exists' + os.path.join(save_folder, all_filename))
            exit(1)

    with open(os.path.join(save_folder, train_filename), 'w') as f:
        print('Creating file... ', os.path.join(save_folder, train_filename))
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(save_folder, validation_filename), 'w') as f:
        print('Creating file... ', os.path.join(save_folder, validation_filename), )
        for item in validation_list:
            f.write("%s\n" % item)
    with open(os.path.join(save_folder, test_filename), 'w') as f:
        print('Creating file... ', os.path.join(save_folder, test_filename))
        for item in test_list:
            f.write("%s\n" % item)
    with open(os.path.join(save_folder, all_filename), 'w') as f:
        print('Creating file... ', os.path.join(save_folder, all_filename))
        for item in type_all_sequences:
            f.write(item[0][0] + ';' + str(item[1]) + '\n')

    print("Finish")

    print("Maybe you want to create the symbolic links....")

    print(
        'while read line; do folder=$(echo $line | cut -d \'/\' -f 1); filenamewithpath=$(echo $line | cut --d \';\' -f 1); filename=$(echo $filenamewithpath | cut --d \'/\' -f 2); echo mkdir -p test/$folder; echo ln -s ../../$filenamewithpath test/$folder/$filename ; done < test_list.txt')


def tokenize(text, token_list=['/', '_', ';', '-'], split=True):
    text_ = text
    for token in token_list:
        text_ = text_.replace(token, '.')
    if split:
        text_ = text_.split('.')
    return text_


def getFrameNumber(extract_field_from_path, frame_filename):
    try:
        # we will try to split the filename with some tokens and then take the frame-number and convert it to
        # integer.
        if extract_field_from_path == -1:
            raise
        current_frame_number = int(tokenize(frame_filename)[extract_field_from_path])
    except:
        print('Current settings does not allow to extract the filename as number. Currently using field: ' + str(
            extract_field_from_path) + '\nExpected a number, but asket to extract field ' +
              str(extract_field_from_path) + ' from the following list. Remember: start to count from zero!')
        print(frame_filename.replace('_', '.', '-').replace('/', '.').split('.'))
        exit()
    return current_frame_number
