import argparse
import multiprocessing
import os
import socket  # to get the machine name
import time
import warnings
from collections import Counter
from datetime import datetime
from functools import partial
from statistics import mean

import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.transforms as transforms
import tqdm
import random
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import SNRDistance, LpDistance, CosineSimilarity
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from dataloaders.sequencedataloader import fromAANETandDualBisenet, fromGeneratedDataset, \
    triplet_BOO, triplet_OBB, kitti360, Kitti2011_RGB, triplet_ROO, triplet_ROO_360, \
    lstm_txt_dataloader, txt_dataloader
from dataloaders.transforms import GenerateBev, Mirror, Normalize, Rescale, ToTensor
from miscellaneous.utils import init_function, send_telegram_message, send_telegram_picture, \
    student_network_pass, svm_generator, svm_testing, covmatrix_generator, mahalanobis_testing, lstm_network_pass, \
    svm_testing_lstm, mahalanobis_testing_lstm, get_distances, get_distances_embb
from model.models import Resnet, LSTM, GRU, VGG, Mobilenet_v3, Inception_v3, Freezed_Model


def str2bool(v):
    """
    Parsing boolean values with argparse
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Args:
        v:

    Returns:

    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(args, dataloader_test, dataloader_train=None, dataloader_val=None, save_embeddings=None, test_path=None):
    print('\n<<<<<<<<<<<<<<<<<< START TESTING >>>>>>>>>>>>>>>>>>')

    loss_val = None

    if args.embedding:
        if args.lossfunction == 'SmoothL1':
            criterion = torch.nn.SmoothL1Loss(reduction='mean')
        elif args.lossfunction == 'L1':
            criterion = torch.nn.L1Loss(reduction='mean')
        elif args.lossfunction == 'MSE':
            criterion = torch.nn.MSELoss(reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.embedding and os.path.isfile(args.centroids_path):
        gt_list = []
        embeddings = np.loadtxt(args.centroids_path, delimiter='\t')
        labels = np.loadtxt(args.label_centroids_path, delimiter='\t')
        for i in range(7):
            gt_list.append(np.mean(embeddings[labels == i], axis=0))
        gt_list = torch.FloatTensor(gt_list)
    else:
        gt_list = None

    # Build model
    # The embeddings should be returned if we are using Techer/Student or triplet loss
    return_embeddings = args.embedding or args.triplet or args.metric

    if 'vgg' in args.model:
        model = VGG(pretrained=args.pretrained, embeddings=return_embeddings, num_classes=args.num_classes,
                    version=args.model)
    elif 'resnet' in args.model:
        model = Resnet(pretrained=args.pretrained, embeddings=return_embeddings, num_classes=args.num_classes,
                       version=args.model)
    elif 'mobilenet' in args.model:
        model = Mobilenet_v3(pretrained=args.pretrained, embeddings=return_embeddings,
                             num_classes=args.num_classes,
                             version=args.model)
    elif 'inception' in args.model:
        model = Inception_v3(pretrained=args.pretrained, embeddings=return_embeddings,
                             num_classes=args.num_classes)
    elif args.model == 'LSTM' or args.model == 'GRU':
        if args.model == 'LSTM':
            model = LSTM(args.num_classes, args.lstm_dropout, args.fc_dropout, embeddings=args.metric,
                         num_layers=args.lstm_layers, input_size=args.lstm_input, hidden_size=args.lstm_hidden)
        else:
            model = GRU(args.num_classes, args.lstm_dropout, args.fc_dropout, embeddings=args.metric,
                        num_layers=args.lstm_layers, input_size=args.lstm_input, hidden_size=args.lstm_hidden)
        if 'resnet' in args.feature_model:
            feature_extractor_model = Resnet(pretrained=False, embeddings=True, version=args.feature_model)
        elif 'vgg' in args.feature_model:
            feature_extractor_model = VGG(pretrained=False, embeddings=True, version=args.feature_model)
        elif 'mobilenet' in args.feature_model:
            feature_extractor_model = Mobilenet_v3(pretrained=False, embeddings=True, version=args.feature_model)
        elif 'inception' in args.feature_model:
            feature_extractor_model = Inception_v3(pretrained=False, embeddings=True)

        # load saved feature extractor model
        if args.feature_detector_path is not None and os.path.isfile(args.feature_detector_path):
            print("=> loading checkpoint '{}'".format(args.feature_detector_path))
            checkpoint = torch.load(args.feature_detector_path, map_location='cpu')
            feature_extractor_model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}'".format(args.feature_detector_path))
        else:
            print("=> no checkpoint found at '{}'".format(args.feature_detector_path))
    else:
        print('Wrong model selection')
        exit(-1)

    if args.freeze and not args.model == 'LSTM':
        print("=> training a fully connected classificator from a metric learning model")
        model = Freezed_Model(model, args.feature_detector_path, args.num_classes, num_layers=1)  # This is hardcoded

    # load Saved Model
    loadpath = args.load_path
    if os.path.isfile(loadpath):
        print("=> Loading checkpoint '{}' ... ".format(loadpath))
        checkpoint = torch.load(loadpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> OK! Checkpoint loaded! '{}'".format(loadpath))
    else:
        print("=> no checkpoint found at '{}'".format(loadpath))

    if torch.cuda.is_available() and args.use_gpu:
        if args.model == 'LSTM':
            model = model.cuda()
            feature_extractor_model = feature_extractor_model.cuda()
            feature_extractor_model.eval()
        else:
            model = model.cuda()

    # Start testing

    # LSTM tests
    if args.model == 'LSTM':
        if not args.metric:
            # lstm con fully-connected to classify
            confusion_matrix, acc_val, loss_val = validation(args, feature_extractor_model, criterion, dataloader_test,
                                                             LSTM=model)
        else:
            # otherwise instead of the fully-connected classifier, we can test
            # with a metric-learning approach, and in this case KEVIN libs were used

            if args.test_method == 'svm':
                classifier = svm_generator(args, feature_extractor_model, dataloader_train=dataloader_train,
                                           dataloader_val=dataloader_val, LSTM=model)
                confusion_matrix, acc_val = svm_testing_lstm(feature_extractor_model, dataloader_test, classifier,
                                                             LSTM=model)
            elif args.test_method == 'mahalanobis':
                covariances = covmatrix_generator(args, feature_extractor_model, dataloader_train=dataloader_train,
                                                  dataloader_val=dataloader_val, LSTM=model)
                confusion_matrix, acc_val = mahalanobis_testing_lstm(feature_extractor_model,
                                                                     dataloader_test, covariances,
                                                                     LSTM=model)
            else:
                # What if not == svm ?
                print('What if not svm or mahalanobis ?')
                exit(-1)

    # RESNETs test
    elif args.triplet:
        # TRIPLET IS OUR OLD CODE , FIRST ATTEMPT
        if args.test_method == 'svm':
            # Generates svm with the last train
            classifier = svm_generator(args, model, dataloader_train=dataloader_train, dataloader_val=dataloader_val)
            confusion_matrix, acc_val, _ = svm_testing(args, model, dataloader_test, classifier)
        elif args.test_method == 'mahalanobis':
            covariances = covmatrix_generator(args, model, dataloader_train=dataloader_train,
                                              dataloader_val=dataloader_val)
            confusion_matrix, acc_val, _ = mahalanobis_testing(args, model, dataloader_test, covariances)
        else:
            print("=> no test method found")
            exit(-1)

    elif args.metric:
        # SIMILAR TO TRIPLET, BUT USING KEVING LIBS

        if args.test_method == 'svm':
            # Generates svm with the last train
            classifier = svm_generator(args, model, dataloader_train=dataloader_train, dataloader_val=dataloader_val)
            if args.get_scores:
                confusion_matrix, acc_val, export_data, scores = svm_testing(args, model, dataloader_test, classifier,
                                                                             probs=True)
            else:
                confusion_matrix, acc_val, export_data = svm_testing(args, model, dataloader_test, classifier)

        elif args.test_method == 'mahalanobis':
            covariances = covmatrix_generator(args, model, dataloader_train=dataloader_train,
                                              dataloader_val=dataloader_val)
            confusion_matrix, acc_val, export_data = mahalanobis_testing(args, model, dataloader_test, covariances)

        elif args.test_method == 'distance':
            distance_array = get_distances(args, dataloader_test, model, gt_list)
            distancespath = './distances'
            if not os.path.isdir(distancespath):
                os.makedirs(distancespath)
            np.save(os.path.join(distancespath, 'distances.npy'), distance_array)

        else:
            print("=> no test method found")
            exit(-1)

        # export_data: this list contains data to create a file that is similar to 'test_list.txt' used
        # in txt_dataloader. will be used to compare RESNET vs LSTM as they are already in 'per-sequence' format.
        if args.export_data:

            # sets filename
            if args.test_method == 'svm':
                filename = '/tmp/' + str(int(time.time())) + '_' + os.path.splitext(os.path.split(test_path)[1])[
                    0] + '_resnet_export_svm' + \
                           os.path.splitext(os.path.split(test_path)[1])[1]
                print('saving data in: ' + filename)
            elif args.test_method == 'mahalanobis':
                filename = '/tmp/' + str(int(time.time())) + '_' + os.path.splitext(os.path.split(test_path)[1])[
                    0] + '_resnet_export_mahalanobis' + \
                           os.path.splitext(os.path.split(test_path)[1])[1]
                print('saving data in: ' + filename)

            # create filename
            with open(filename, "w") as output:
                for i in range(len(export_data[0])):
                    line = export_data[0][i] + ';' + str(export_data[1][i]) + ';' + str(export_data[2][i]) + '\n'
                    output.write(line)

        if args.get_scores:
            # write scores in a log
            scorespath = './scores'
            if not os.path.isdir(scorespath):
                os.makedirs(scorespath)
            np.savez(os.path.join(scorespath, 'scores.npz'), prob=scores[0], logit=scores[1])

    else:
        # THIS IS OUR BASELINE, WITHOUT TRIPLET FLAVOURS (OURS OR KEVIN)
        if args.test_method == 'distance':
            save_embeddings = True
        confusion_matrix, acc_val, _ = validation(args, model, criterion, dataloader_test, gt_list=gt_list,
                                                  save_embeddings=save_embeddings)

    if confusion_matrix is not None:
        plt.figure(figsize=(10, 7))
        title = str(socket.gethostname()) + '\nTEST '
        plt.title(title)
        if args.norm_conf_matrix == 'index' or args.norm_conf_matrix == 'columns':
            sn.heatmap(confusion_matrix, annot=True, fmt='.2f')
        else:
            sn.heatmap(confusion_matrix, annot=True, fmt='.1f')
        if args.telegram:
            if loss_val is not None:
                send_telegram_picture(plt, "TEST" + "\nacc_val: " + str(acc_val) + "\nloss_val: " + str(loss_val))
            else:
                send_telegram_picture(plt, "TEST" + "\nacc_val: " + str(acc_val))

        if not args.nowandb:
            wandb.log({"Test/Acc": acc_val, "conf-matrix_test": wandb.Image(plt)})


def validation(args, model, criterion, dataloader, gt_list=None, weights=None,
               save_embeddings=None, miner=None, acc_metric=None, LSTM=None):
    """

    This function is called both from actual 'validation' and during the 'test'.
    Save embeddings to disk in a similar way of teacher_train.py. Useful in testing

    Args:
        miner:
        acc_metric:
        args:
        model:
        criterion:
        dataloader:
        classifier:
        gt_list:
        weights:
        save_embeddings: PATH+FILENAME (FILEPATH) /.../name.txt ; save the embeddings here

    Returns:
        depends...

        not args.triplet: -> conf_matrix, acc, loss_val_mean
            args.triplet: -> None       , acc, loss_val_mean


    """
    print('\n>>>>>>>>>>>>>>>>>> START VALIDATION <<<<<<<<<<<<<<<<<<')

    loss_record = 0.0
    if args.metric:
        acc_record = {}
    else:
        acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    # if save_embeddings, this will be populated
    if save_embeddings:
        all_embedding_matrix = []

    with torch.no_grad():

        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('Validation... ')

        for sample in dataloader:

            embedding = None

            if args.model == 'LSTM':
                LSTM.eval()
                acc, loss, label, predict = lstm_network_pass(args, sample, criterion, model, LSTM, miner=miner,
                                                              acc_metric=acc_metric)
            else:
                model.eval()
                acc, loss, label, predict, embedding , _ = student_network_pass(args, sample, criterion, model,
                                                                            gt_list=gt_list, weights_param=weights,
                                                                            miner=miner, acc_metric=acc_metric,
                                                                            return_embedding=save_embeddings)

            if embedding is not None:
                all_embedding_matrix.append(embedding)

            if label is not None and predict is not None:
                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)

            loss_record += loss.item()

            if args.metric:
                acc_record = dict(Counter(acc_record) + Counter(acc))
            else:
                acc_record += acc

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

        tq.close()

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader)
    print('loss for test/validation : %f' % loss_val_mean)

    if args.metric:
        acc = mean(acc_record[k] for k in acc_record) / len(dataloader)
        print('Accuracy for test/validation : %f\n' % acc)
        acc_record = {k: v / len(dataloader) for k, v in acc_record.items()}
        acc = acc_record
    else:
        acc = acc_record / len(dataloader)
        print('Accuracy for test/validation : %f\n' % acc)

    if save_embeddings and not args.test_method == 'distance':
        all_embedding_matrix = np.asarray(all_embedding_matrix)
        if not os.path.isdir(args.saveEmbeddingsPath):
            os.makedirs(args.saveEmbeddingsPath)
        np.savetxt(os.path.join(args.saveEmbeddingsPath, 'embeddings.txt'), np.vstack(all_embedding_matrix),
                   delimiter='\t', fmt='%s')
        np.savetxt(os.path.join(args.saveEmbeddingsPath, 'labels.txt'), labelRecord, delimiter='\t', fmt='%s')

    if args.test_method == 'distance':
        distancespath = './distances'
        if not os.path.isdir(distancespath):
            os.makedirs(distancespath)
        embbeding = np.vstack(all_embedding_matrix)
        distances = get_distances_embb(embbeding, gt_list)
        np.save(os.path.join(distancespath, 'distances.npy'), distances)

    if args.get_scores:
        # write scores in a log
        scorespath = './scores'
        if not os.path.isdir(scorespath):
            os.makedirs(scorespath)
        scores = np.vstack(all_embedding_matrix)
        np.save(os.path.join(scorespath, 'scores.npy'), scores)

    if labelRecord.size != 0 and predRecord.size != 0:
        conf_matrix = pd.crosstab(labelRecord, predRecord, rownames=['Actual'], colnames=['Predicted'],
                                  normalize=args.norm_conf_matrix)
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0.0)
    else:
        conf_matrix = None

    return conf_matrix, acc, loss_val_mean


def train(args, model, optimizer, scheduler, dataloader_train, dataloader_val, valfolder, GLOBAL_EPOCH, LSTM=None):
    """

    Do the training. The LOSS depends on the value of
        weighted    : standard classifier with weighted classes
        embedding   : student-case
        triple      : BOO and OBB
        .. else ..  : standard classifier

    Args:
        args: from the main, all the args
        model:
        optimizer:
        dataloader_train:
        dataloader_val:
        acc_pre:
        valfolder:
        GLOBAL_EPOCH:
        gtmodel:

    Returns:

    """
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    max_val_acc = 0.0
    min_val_loss = np.inf

    if args.embedding:  # For Teacher/Student training
        miner = None  # No need of miner
        acc_metric = None  # No nedd of metric acc
        # Build loss criterion
        if args.weighted:
            if args.weight_tensor == 'Kitti360':
                weights = [0.99, 1.01, 0.98, 0.99, 1.05, 0.98, 0.99]
            elif args.weight_tensor == 'Alcala':
                weights = [0.89, 1.13, 1.09, 1.05, 0.93, 1.06, 0.86]
            elif args.weight_tensor == 'Kitti2011':
                weights = [1.06, 1.11, 1.12, 0.98, 0.99, 0.96, 0.78]

            if args.lossfunction == 'SmoothL1':
                criterion = torch.nn.SmoothL1Loss(reduction='none')
            elif args.lossfunction == 'L1':
                criterion = torch.nn.L1Loss(reduction='none')
            elif args.lossfunction == 'MSE':
                criterion = torch.nn.MSELoss(reduction='none')
        else:
            weights = None
            if args.lossfunction == 'SmoothL1':
                criterion = torch.nn.SmoothL1Loss(reduction='mean')
            elif args.lossfunction == 'L1':
                criterion = torch.nn.L1Loss(reduction='mean')
            elif args.lossfunction == 'MSE':
                criterion = torch.nn.MSELoss(reduction='mean')

        # Build gt centroids to measure distances
        gt_list = []
        embeddings = np.loadtxt(args.centroids_path, delimiter='\t')
        splits = np.array_split(embeddings, 7)
        for i in range(7):
            gt_list.append((np.mean(splits[i], axis=0)))
        gt_list = torch.FloatTensor(gt_list)

    elif args.triplet or args.lossfunction == 'triplet':
        gt_list = None  # No need of centroids
        miner = None  # No need of miner
        acc_metric = None  # No nedd of metric acc
        # Build loss criterion
        if args.weighted:
            if args.weight_tensor == 'Kitti360':
                weights = [0.99, 1.01, 0.98, 0.99, 1.05, 0.98, 0.99]
            elif args.weight_tensor == 'Alcala':
                weights = [0.89, 1.13, 1.09, 1.05, 0.93, 1.06, 0.86]
            elif args.weight_tensor == 'Kitti2011':
                weights = [1.06, 1.11, 1.12, 0.98, 0.99, 0.96, 0.78]

            if args.distance_function == 'pairwise':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=torch.nn.PairwiseDistance(p=args.p),
                    margin=args.margin, reduction='none')
            elif args.distance_function == 'cosine':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=lambda x, y: 1.0 - cosine_similarity(x, y),
                    margin=args.margin, reduction='none')
            elif args.distance_function == 'SNR':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=lambda x, y: torch.var(x - y) / torch.var(x),
                    margin=args.margin, reduction='none')
            else:
                criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=args.p, reduction='none')

        else:
            weights = None
            if args.distance_function == 'pairwise':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=torch.nn.PairwiseDistance(p=args.p),
                    margin=args.margin, reduction='mean')

            elif args.distance_function == 'cosine':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=lambda x, y: 1.0 - cosine_similarity(x, y),
                    margin=args.margin, reduction='mean')

            elif args.distance_function == 'SNR':
                criterion = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=lambda x, y: torch.var(x - y) / torch.var(x),
                    margin=args.margin, reduction='mean')
            else:
                criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=args.p, reduction='mean')

    elif args.metric:
        gt_list = None  # No need of centroids
        # Accuracy calculator for metric learning
        acc_metric = AccuracyCalculator(exclude=('AMI', 'NMI'), avg_of_avgs=True)
        # Accuracy metrics for metric learning
        if args.weighted:
            if args.weight_tensor == 'Kitti360':
                weights = [0.99, 1.01, 0.98, 0.99, 1.05, 0.98, 0.99]
                class_weights = torch.FloatTensor(weights).cuda()
            elif args.weight_tensor == 'Alcala':
                weights = [0.89, 1.13, 1.09, 1.05, 0.93, 1.06, 0.86]
                class_weights = torch.FloatTensor(weights).cuda()
            elif args.weight_tensor == 'Kitti2011':
                weights = [1.06, 1.11, 1.12, 0.98, 0.99, 0.96, 0.78]
                class_weights = torch.FloatTensor(weights).cuda()
            reducer = reducers.ClassWeightedReducer(class_weights)
        elif args.nonzero:
            weights = None
            reducer = reducers.AvgNonZeroReducer()
        else:
            weights = None
            reducer = reducers.MeanReducer()

        if args.distance_function == 'SNR':
            criterion = losses.TripletMarginLoss(margin=args.margin, swap=False, smooth_loss=False,
                                                 triplets_per_anchor="all",
                                                 distance=SNRDistance(normalize_embeddings=args.normalize),
                                                 reducer=reducer)
            if args.miner:
                miner = miners.TripletMarginMiner(margin=args.margin * 2.0,
                                                  type_of_triplets=args.TripletMarginMinerType,
                                                  distance=SNRDistance(normalize_embeddings=args.normalize))
            else:
                miner = None

        elif args.distance_function == 'pairwise':
            criterion = losses.TripletMarginLoss(margin=args.margin, swap=True, smooth_loss=False,
                                                 # triplets_per_anchor="all",
                                                 triplets_per_anchor=100,
                                                 distance=LpDistance(p=args.p, normalize_embeddings=args.normalize),
                                                 reducer=reducer)
            if args.miner:
                miner = miners.TripletMarginMiner(margin=args.margin_miner, # * 2.0,
                                                  type_of_triplets=args.TripletMarginMinerType,
                                                  distance=LpDistance(p=args.p, normalize_embeddings=args.normalize))
            else:
                miner = None

        elif args.distance_function == 'cosine':
            criterion = losses.TripletMarginLoss(margin=args.margin, swap=False, smooth_loss=False,
                                                 triplets_per_anchor="all",
                                                 distance=CosineSimilarity(),
                                                 reducer=reducer)
            if args.miner:
                miner = miners.TripletMarginMiner(margin=args.margin * 2.0,
                                                  type_of_triplets=args.TripletMarginMinerType,
                                                  distance=CosineSimilarity())
            else:
                miner = None

    else:
        gt_list = None  # No need of centroids
        miner = None  # No need of miner
        acc_metric = None
        if args.lossfunction == 'focal':
            weights = None
            kwargs = {"alpha": 0.5, "gamma": 5.0, "reduction": 'mean'}
            criterion = kornia.losses.FocalLoss(**kwargs)
        else:
            if args.weighted:
                if args.weight_tensor == 'Kitti360':
                    weights = [0.99, 1.01, 0.98, 0.99, 1.05, 0.98, 0.99]
                    class_weights = torch.FloatTensor(weights).cuda()
                elif args.weight_tensor == 'Alcala':
                    weights = [0.89, 1.13, 1.09, 1.05, 0.93, 1.06, 0.86]
                    class_weights = torch.FloatTensor(weights).cuda()
                elif args.weight_tensor == 'Kitti2011':
                    weights = [1.06, 1.11, 1.12, 0.98, 0.99, 0.96, 0.78]
                    class_weights = torch.FloatTensor(weights).cuda()
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                weights = None
                criterion = torch.nn.CrossEntropyLoss()

    if args.model == 'LSTM':
        model.eval()
        LSTM.train()
    else:
        model.train()

    if not args.nowandb:  # if nowandb flag was set, skip
        if args.model == 'LSTM':
            wandb.watch(LSTM, log="all")
        else:
            wandb.watch(model, log="all")

    current_batch = 0
    patience = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        print("\n\n===========================================================")
        print("date and time:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), '\n')
        if GLOBAL_EPOCH is not None:
            with GLOBAL_EPOCH.get_lock():
                GLOBAL_EPOCH.value = epoch
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %.e' % (epoch, lr))
        loss_record = 0.0
        if args.metric:
            acc_record = {}
        else:
            acc_record = 0.0

        for sample in dataloader_train:
            if args.model == 'LSTM':
                acc, loss, _, _ = lstm_network_pass(args, sample, criterion, model, LSTM, miner=miner,
                                                    acc_metric=acc_metric)
            else:
                acc, loss, _, _, _ , hard_triplets = student_network_pass(args, sample, criterion, model, gt_list=gt_list,
                                                          weights_param=weights, miner=miner, acc_metric=acc_metric)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()

            if args.metric:
                acc_record = dict(Counter(acc_record) + Counter(acc))
                acc = mean(acc[k] for k in acc)
            else:
                acc_record += acc

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.log({"batch_training_accuracy": acc,
                           "batch_training_loss": loss.item(),
                           "batch_current_batch": current_batch,
                           "batch_current_epoch": epoch,
                           "batch_hardtriplets": hard_triplets})

                current_batch += 1

        tq.close()

        # ReduceLROnPlateau is set after the validation block
        if args.scheduler and args.scheduler_type == 'MultiStepLR':
            print("MultiStepLR step call")
            scheduler.step()

        # Calculate metrics
        loss_train_mean = loss_record / len(dataloader_train)
        print('...')
        print('loss for train : {:.4f} - Total elements: {:2d}'.format(loss_train_mean, len(dataloader_train)))

        if args.metric:
            acc_train = mean(acc_record[k] for k in acc_record)
            acc_train = acc_train / len(dataloader_train)
            print('acc for train : %f' % acc_train)

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.log({"Train/loss": loss_train_mean,
                           "Train/MAP": acc_record['mean_average_precision'] / len(dataloader_train),
                           "Train/MAPR": acc_record['mean_average_precision_at_r'] / len(dataloader_train),
                           "Train/PA1": acc_record['precision_at_1'] / len(dataloader_train),
                           "Train/Rp": acc_record['r_precision'] / len(dataloader_train),
                           "Train/lr": optimizer.param_groups[0]['lr'],
                           "Completed epoch": epoch})

        else:
            acc_train = acc_record / len(dataloader_train)
            print('acc for train : %f' % acc_train)

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.log({"Train/loss": loss_train_mean,
                           "Train/acc": acc_train,
                           "Train/lr": optimizer.param_groups[0]['lr'],
                           "Completed epoch": epoch})

        if epoch % args.validation_step == 0:
            if args.model == 'LSTM':
                confusion_matrix, acc_val, loss_val = validation(args, model, criterion, dataloader_val, LSTM=LSTM,
                                                                 miner=miner, acc_metric=acc_metric)
                LSTM.train()
            else:
                confusion_matrix, acc_val, loss_val = validation(args, model, criterion, dataloader_val,
                                                                 gt_list=gt_list,
                                                                 weights=weights, miner=miner, acc_metric=acc_metric)
                model.train()

            if args.scheduler and args.scheduler_type == 'ReduceLROnPlateau':
                print("ReduceLROnPlateau step call")
                scheduler.step(loss_val)

            if confusion_matrix is not None:
                plt.figure(figsize=(10, 7))
                title = str(socket.gethostname()) + '\nEpoch: ' + str(epoch)
                plt.title(title)
                sn.heatmap(confusion_matrix, annot=True, fmt='.3f')

            if args.telegram and confusion_matrix is not None:
                send_telegram_picture(plt,
                                      "Epoch: " + str(epoch) +
                                      "\nLR: " + str(optimizer.param_groups[0]['lr']) +
                                      "\nacc_val: " + str(acc_val) +
                                      "\nloss_val: " + str(loss_val))

            if not args.nowandb and confusion_matrix is not None:  # if nowandb flag was set, skip
                wandb.log({"Val/loss": loss_val,
                           "Val/Acc": acc_val,
                           "Completed epoch": epoch,
                           "conf-matrix_{}_{}".format(valfolder, epoch): wandb.Image(plt)})

            elif not args.nowandb and args.triplet:
                wandb.log({"Val/loss": loss_val,
                           "Val/Acc": acc_val,
                           "Completed epoch": epoch})

            elif not args.nowandb and args.metric:
                wandb.log({"Val/loss": loss_val,
                           "Val/MAP": acc_val['mean_average_precision'],
                           "Val/MAPR": acc_val['mean_average_precision_at_r'],
                           "Val/PA1": acc_val['precision_at_1'],
                           "Val/Rp": acc_val['r_precision'],
                           "Completed epoch": epoch})
            if args.metric:
                acc_val = acc_val['mean_average_precision_at_r']

            if (max_val_acc < acc_val) or (min_val_loss > loss_val):
                patience = 0

                if max_val_acc < acc_val:
                    max_val_acc = acc_val
                    print('Best global accuracy: {}'.format(max_val_acc))
                    wandb.run.summary["Best_Accuracy"] = max_val_acc
                if min_val_loss > loss_val:
                    min_val_loss = loss_val
                    print('Best global loss: {}'.format(min_val_loss))

                if args.savemodel:
                    if args.nowandb:
                        loadpath = os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                                 args.model,
                                                                                                 epoch))
                        if args.model == 'LSTM':
                            print('Saving model: ', loadpath)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': LSTM.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
                                'loss': loss,
                            }, os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                             args.model, epoch)))
                        else:
                            print('Saving model: ', loadpath)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
                                'loss': loss,
                            }, os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                             args.model, epoch)))
                    else:
                        loadpath = os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                                 wandb.run.id, epoch))
                        if args.model == 'LSTM':
                            print('Saving model: ', os.path.join(loadpath))
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': LSTM.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
                                'loss': loss,
                            }, os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                             wandb.run.id, epoch)))
                        else:
                            print('Saving model: ', os.path.join(loadpath))
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
                                'loss': loss,
                            }, os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                             wandb.run.id, epoch)))

            elif epoch < args.patience_start:
                patience = 0

            else:
                patience += 1

        if patience >= args.patience > 0:
            break


def main(args, model=None):
    # Try to avoid randomness -- https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.seed = checkpoint['epoch']

    GLOBAL_EPOCH = multiprocessing.Value('i', args.seed)
    seed = multiprocessing.Value('i', args.seed)

    init_fn = partial(init_function, seed=seed, epoch=GLOBAL_EPOCH)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # init_fn = None
    # GLOBAL_EPOCH = None

    # workaround for "TOO MANY OPEN FILES"
    # https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open
    torch.multiprocessing.set_sharing_strategy('file_system')
    #
    # a faster workaround is to change the ulimits in linux, per shell based
    # https://stackoverflow.com/questions/16526783/python-subprocess-too-many-open-files
    #
    # check the limits   > ulimit -a
    #         core file size          (blocks, -c) 0
    #         data seg size           (kbytes, -d) unlimited
    #         scheduling priority             (-e) 0
    #         file size               (blocks, -f) unlimited
    #         pending signals                 (-i) 124997
    #         max locked memory       (kbytes, -l) 65536
    #         max memory size         (kbytes, -m) unlimited
    #         open files                      (-n) 1024             <<<<<<< this is the issue
    #         pipe size            (512 bytes, -p) 8
    #         POSIX message queues     (bytes, -q) 819200
    #         real-time priority              (-r) 0
    #         stack size              (kbytes, -s) 8192
    #         cpu time               (seconds, -t) unlimited
    #         max user processes              (-u) 124997
    #         virtual memory          (kbytes, -v) unlimited
    #         file locks                      (-x) unlimited
    #
    # change the limits  > ulimit -Sn 10000
    #

    # create dataset and dataloader
    data_path = args.dataset

    # TODO: ALVARO! Esto es lo que queria editar un poco para que quede claro cuando se usa uno y el otro, a lo mejor no con if elif else pero simples if..
    if args.dataloader == 'lstm_txt_dataloader' or args.dataloader == 'txt_dataloader':

        # args.dataset          >>> *always used*
        #                           it acts as dataset_train in the case you want
        #                           to specify different train/val/test folders

        # IF you want to use other validation/test folders, specify them using:
        # args.dataset_val      >>> path to the folder
        # args.dataset_test     >>> path to the folder

        if all(map(os.path.isfile, args.dataset)) and all(map(os.path.isfile, args.dataset_val)) and all(
                map(os.path.isfile, args.dataset_test)):
            train_path = args.dataset  # Path list to train dataset
            val_path = args.dataset_val  # Path list to validation dataset
            test_path = args.dataset_test  # Path list to test dataset
        else:
            assert os.path.isfile(os.path.join(args.dataset[0], 'train/train_list.txt')), "Error in train dataset"
            assert os.path.isfile(
                os.path.join(args.dataset[0], 'validation/validation_list.txt')), "Error in validation dataset"
            assert os.path.isfile(os.path.join(args.dataset[0], 'test/test_list.txt')), "Error in test dataset"
            train_path = os.path.join(args.dataset[0], 'train/train_list.txt')
            val_path = os.path.join(args.dataset[0], 'validation/validation_list.txt')
            test_path = os.path.join(args.dataset[0], 'test/test_list.txt')

            # for some reason, in some cases, if we're using lstm_txt_dataloader than test_path is not in the args.xxxx
            # instead we use data_path, or data_path = args.dataset previously defined.
            if not args.dataset_test:
                args.dataset_test = test_path
            if not args.dataset_val:
                args.dataset_val = val_path

    elif '360' not in args.dataloader:
        # All sequence folders
        folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                            os.path.isdir(os.path.join(data_path, folder))])

        # Exclude test samples
        folders = folders[folders != os.path.join(data_path, '2011_09_30_drive_0028_sync')]
        test_path = os.path.join(data_path, '2011_09_30_drive_0028_sync')

        # Exclude validation samples"
        train_path = folders[folders != os.path.join(data_path, '2011_10_03_drive_0034_sync')]
        val_path = os.path.join(data_path, '2011_10_03_drive_0034_sync')

    else:
        # THIS ARE THE SECUENCIES FOR KITTI360
        train_sequence_list = ['2013_05_28_drive_0003_sync',
                               '2013_05_28_drive_0002_sync',
                               '2013_05_28_drive_0005_sync',
                               '2013_05_28_drive_0006_sync',
                               '2013_05_28_drive_0007_sync',
                               '2013_05_28_drive_0009_sync',
                               '2013_05_28_drive_0010_sync']
        val_sequence_list = ['2013_05_28_drive_0004_sync']
        test_sequence_list = ['2013_05_28_drive_0000_sync']

        # This are the sequences for testing a train with kitt2011 with kitti360
        kitti360_sequence_list = ['2013_05_28_drive_0003_sync',
                                  '2013_05_28_drive_0002_sync',
                                  '2013_05_28_drive_0005_sync',
                                  '2013_05_28_drive_0006_sync',
                                  '2013_05_28_drive_0007_sync',
                                  '2013_05_28_drive_0009_sync',
                                  '2013_05_28_drive_0010_sync',
                                  '2013_05_28_drive_0004_sync',
                                  '2013_05_28_drive_0000_sync']

    if args.model == 'inception_v3':
        img_rescale = transforms.Resize((299, 299))
    else:
        img_rescale = transforms.Resize(args.image_size)

    aanetTransforms = transforms.Compose(
        [GenerateBev(decimate=args.decimate), Mirror(), Rescale((224, 224)), Normalize(), ToTensor()])

    # Transforms for OSM in Triplet_OBB and Triplet_BOO dataloaders
    osmTransforms = transforms.Compose(
        [transforms.ToPILImage(), img_rescale, transforms.ToTensor()])

    # Transforms for RGB images (RGB // Homography)
    # rgb_image_train_transforms = transforms.Compose(
    #     [img_rescale, transforms.RandomAffine(15, translate=(0.0, 0.1), shear=(-5, 5)),
    #      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), transforms.ToTensor(),
    #      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    # rgb_image_test_transforms = transforms.Compose([img_rescale, transforms.ToTensor(),
    #                                                 transforms.Normalize((0.485, 0.456, 0.406),
    #                                                                      (0.229, 0.224, 0.225))])

    # Transforms for RGB images (RGB // Homography)
    rgb_image_train_transforms = transforms.Compose(
        [img_rescale, transforms.RandomAffine(15, translate=(0.0, 0.1), shear=(-5, 5)),
         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    rgb_image_test_transforms = transforms.Compose([img_rescale, transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                         (0.229, 0.224, 0.225))])

    # Transforms for Three-dimensional images (The DA was made offline)
    threedimensional_transfomrs = transforms.Compose([img_rescale, transforms.ToTensor()])

    GAN_transfomrs = transforms.Compose([img_rescale, transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))])

    if args.train or (args.test and (args.triplet or args.metric)):

        if not args.nowandb and args.train:  # if nowandb flag was set, skip
            if args.wandb_resume:
                print('Resuming WANDB run, this run will log into: ', args.wandb_resume)
                wandb.init(project=args.project, group=group_id, entity='chiringuito',
                           job_type="training", reinit=True, resume=args.wandb_resume)

                wandb.config.update(args, allow_val_change=True)
            else:
                wandb.init(project=args.project, group=group_id, entity='chiringuito',
                           job_type="training", reinit=True)
                wandb.config.update(args)

        # The dataloaders that not use Kitti360 uses list-like inputs
        if '360' not in args.dataloader and (
                args.dataloader != 'lstm_txt_dataloader' and args.dataloader != 'txt_dataloader'):
            train_path = np.array(train_path)
            val_path = np.array([val_path])

        if args.dataloader == 'Kitti360':  # Used in kitti360 RGB // Homography
            train_dataset = kitti360(args.dataset, train_sequence_list, transform=rgb_image_train_transforms)
            val_dataset = kitti360(args.dataset, val_sequence_list, transform=rgb_image_train_transforms)

        elif args.dataloader == 'Kitti360_3D':  # Used in Kitti360 3D
            train_dataset = kitti360(args.dataset, train_sequence_list, transform=threedimensional_transfomrs)
            val_dataset = kitti360(args.dataset, val_sequence_list, transform=threedimensional_transfomrs)

        elif args.dataloader == "fromAANETandDualBisenet":  # Used in kitti2011 online generated masked 3D images (Deprecated)
            val_dataset = fromAANETandDualBisenet(val_path, args.distance, transform=aanetTransforms)
            train_dataset = fromAANETandDualBisenet(train_path, args.distance, transform=aanetTransforms)

        elif args.dataloader == "generatedDataset":  # Used in Kitti2011 // 3D // Masked 3D
            val_dataset = fromGeneratedDataset(val_path, args.distance, transform=threedimensional_transfomrs,
                                               loadlist=False,
                                               decimateStep=args.decimate,
                                               addGeneratedOSM=False)
            train_dataset = fromGeneratedDataset(train_path, args.distance, transform=threedimensional_transfomrs,
                                                 loadlist=False,
                                                 decimateStep=args.decimate,
                                                 addGeneratedOSM=False)

        elif args.dataloader == "triplet_OBB":  # Used in Kitti2011 Masked 3D // 3D (Not Replicated)

            train_dataset = triplet_OBB(train_path, args.distance, elements=args.num_elements_OBB, canonical=False,
                                        transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs,
                                        loadlist=False)

            val_dataset = triplet_OBB(val_path, args.distance, elements=args.num_elements_OBB, canonical=False,
                                      transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs,
                                      loadlist=False)

        elif args.dataloader == "triplet_BOO":  # Used in Kitti2011 Masked 3D // 3D

            val_dataset = triplet_BOO(val_path, args.distance, canonical=False,
                                      transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs,
                                      decimateStep=args.decimate)

            train_dataset = triplet_BOO(train_path, args.distance, canonical=False,
                                        transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs,
                                        decimateStep=args.decimate)

        elif args.dataloader == "triplet_ROO":  # Used in Kitti2011 RGB // Homograpy

            val_dataset = triplet_ROO(val_path, transform_osm=osmTransforms, transform_rgb=rgb_image_train_transforms)

            train_dataset = triplet_ROO(train_path, transform_osm=osmTransforms,
                                        transform_rgb=rgb_image_train_transforms)

        elif args.dataloader == "triplet_ROO_360":  # Used in Kitti360 RGB // Homograpy

            val_dataset = triplet_ROO_360(args.dataset, val_sequence_list, transform_osm=osmTransforms,
                                          transform_rgb=rgb_image_train_transforms)

            train_dataset = triplet_ROO_360(args.dataset, train_sequence_list, transform_osm=osmTransforms,
                                            transform_rgb=rgb_image_train_transforms)

        elif args.dataloader == "triplet_3DOO_360":  # Used in Kitti360 3D

            val_dataset = triplet_ROO_360(args.dataset, val_sequence_list, transform_osm=osmTransforms,
                                          transform_3d=threedimensional_transfomrs)

            train_dataset = triplet_ROO_360(args.dataset, train_sequence_list, transform_osm=osmTransforms,
                                            transform_3d=threedimensional_transfomrs)

        elif args.dataloader == "Kitti2011_RGB":  # Used in Kitti2011 // RGB // Homography

            val_dataset = Kitti2011_RGB(val_path, transform=rgb_image_test_transforms)

            train_dataset = Kitti2011_RGB(train_path, transform=rgb_image_train_transforms)

        elif args.dataloader == 'lstm_txt_dataloader' and not all(
                map(lambda x: '3D' in x, train_path)):  # // RGB // Homography
            print('Training with rgb Data augmentation')
            val_dataset = lstm_txt_dataloader(val_path, transform=rgb_image_train_transforms,
                                              all_in_ram=args.all_in_ram, fixed_lenght=args.fixed_length)
            train_dataset = lstm_txt_dataloader(train_path, transform=rgb_image_train_transforms,
                                                all_in_ram=args.all_in_ram, fixed_lenght=args.fixed_length)

        elif args.dataloader == 'lstm_txt_dataloader':  # // 3D // 3D-Masked
            print('Training with three-dimensional Data')
            val_dataset = lstm_txt_dataloader(val_path, transform=threedimensional_transfomrs,
                                              all_in_ram=args.all_in_ram, fixed_lenght=args.fixed_length)
            train_dataset = lstm_txt_dataloader(train_path, transform=threedimensional_transfomrs,
                                                all_in_ram=args.all_in_ram, fixed_lenght=args.fixed_length)

        elif args.dataloader == 'txt_dataloader' and not all(
                map(lambda x: '3D' in x, train_path)):  # // RGB // Homography
            print('Training with rgb Data augmentation')
            val_dataset = txt_dataloader(val_path, transform=rgb_image_test_transforms, decimateStep=args.decimate)

            train_dataset = txt_dataloader(train_path, transform=rgb_image_train_transforms, decimateStep=args.decimate)

        elif args.dataloader == 'txt_dataloader':  # // 3D // 3D-Masked
            print('Training with three-dimensional Data')
            val_dataset = txt_dataloader(val_path, transform=threedimensional_transfomrs, decimateStep=args.decimate)

            train_dataset = txt_dataloader(train_path, transform=threedimensional_transfomrs,
                                           decimateStep=args.decimate)

        else:
            raise Exception("Dataloader not found")

        # TODO: not sure if this need to remain or is only for the LSTM.
        # error: each element in list of batch should be of equal size
        # solution found here: https://github.com/pytorch/vision/issues/2624
        if train_dataset.getIsSequence():
            print("Using a sequence dataset ...")
            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=False,
                                          collate_fn=lambda x: x)
        else:
            print("Using a single-frame dataset (not a sequence) ...")
            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=True)

        if val_dataset.getIsSequence():
            dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=False,
                                        collate_fn=lambda x: x)
        else:
            dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=True)

        # we can change the min-elements of each dataloader here
        if args.dataloader == 'lstm_txt_dataloader' and args.usesmallest:
            # check whether to use a fixed number or the min. value within train/val dataset splits.
            if args.defaultsequencelength > 0:
                print("Using --defaultsequencelength: ", str(args.defaultsequencelength))
                smallest = args.defaultsequencelength
            else:
                print("Evaluating min-sequence frame value...")
                smallest = min(train_dataset.min_elements, val_dataset.min_elements)
            print("Between the TRAIN and VALIDATION data, the SMALLEST sequence has ", str(smallest), ' frames')
            train_dataset.min_elements = smallest
            val_dataset.min_elements = smallest

        if args.train:
            # Build model
            # The embeddings should be returned if we are using Techer/Student or triplet loss
            return_embeddings = args.embedding or args.triplet or args.metric or args.freeze

            if 'vgg' in args.model:
                model = VGG(pretrained=args.pretrained, embeddings=return_embeddings, num_classes=args.num_classes,
                            version=args.model)
            elif 'resnet' in args.model:
                model = Resnet(pretrained=args.pretrained, embeddings=return_embeddings, num_classes=args.num_classes,
                               version=args.model)
            elif 'mobilenet' in args.model:
                model = Mobilenet_v3(pretrained=args.pretrained, embeddings=return_embeddings,
                                     num_classes=args.num_classes,
                                     version=args.model)
            elif 'inception' in args.model:
                model = Inception_v3(pretrained=args.pretrained, embeddings=return_embeddings,
                                     num_classes=args.num_classes)
            elif args.model == 'LSTM' or args.model == 'GRU':
                if args.model == 'LSTM':
                    model = LSTM(args.num_classes, args.lstm_dropout, args.fc_dropout, embeddings=args.metric,
                                 num_layers=args.lstm_layers, input_size=args.lstm_input, hidden_size=args.lstm_hidden)
                else:
                    model = GRU(args.num_classes, args.lstm_dropout, args.fc_dropout, embeddings=args.metric,
                                num_layers=args.lstm_layers, input_size=args.lstm_input, hidden_size=args.lstm_hidden)
                if 'resnet' in args.feature_model:
                    feature_extractor_model = Resnet(pretrained=False, embeddings=True, version=args.feature_model)
                elif 'vgg' in args.feature_model:
                    feature_extractor_model = VGG(pretrained=False, embeddings=True, version=args.feature_model)
                elif 'mobilenet' in args.feature_model:
                    feature_extractor_model = Mobilenet_v3(pretrained=False, embeddings=True,
                                                           version=args.feature_model)
                elif 'inception' in args.feature_model:
                    feature_extractor_model = Inception_v3(pretrained=False, embeddings=True)

                # load saved feature extractor model
                if args.feature_detector_path is not None and os.path.isfile(args.feature_detector_path):
                    print("=> loading checkpoint '{}'".format(args.feature_detector_path))
                    checkpoint = torch.load(args.feature_detector_path, map_location='cpu')
                    feature_extractor_model.load_state_dict(checkpoint['model_state_dict'])
                    print("=> loaded checkpoint '{}'".format(args.feature_detector_path))
                else:
                    print("=> no checkpoint found at '{}'".format(args.feature_detector_path))
                    print("=> training all from scratch")

                if torch.cuda.is_available() and args.use_gpu:
                    feature_extractor_model = feature_extractor_model.cuda()
            else:
                print('Wrong model selection')
                exit(-1)

            if args.freeze and not args.model == 'LSTM':
                print("=> training a fully connected classificator from a metric learning model")
                model = Freezed_Model(model, args.feature_detector_path, args.num_classes,
                                      num_layers=1)  # This is hardcoded

            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume, map_location='cpu')
                    args.start_epoch = checkpoint['epoch'] + 1
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if torch.cuda.is_available() and args.use_gpu:
                        model = model.cuda()
                    print("=> loaded checkpoint '{}' (epoch {}) (loss {})"
                          .format(args.resume, checkpoint['epoch'], checkpoint['loss']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
            else:
                if torch.cuda.is_available() and args.use_gpu:
                    model = model.cuda()

            # Build optimizer
            if args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=args.momentum)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            elif args.optimizer == 'adam':
                # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
                optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.adam_weight_decay)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            elif args.optimizer == 'ASGD':
                optimizer = torch.optim.ASGD(model.parameters(), args.lr)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            elif args.optimizer == 'Adamax':
                optimizer = torch.optim.Adamax(model.parameters(), args.lr)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            elif args.optimizer == 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.adam_weight_decay)
                if args.resume:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Set new learning rate from command line
                    optimizer.param_groups[0]['lr'] = args.lr
            else:
                print('not supported optimizer \n')
                exit()

            # Build scheduler
            if args.scheduler:
                if args.resume:
                    param_epoch = checkpoint['epoch']
                else:
                    param_epoch = -1

                if args.scheduler_type == 'MultiStepLR':
                    print("Creating MultiStepLR optimizer with last_epoch: {}".format(param_epoch))
                    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 18, 20], gamma=0.5,
                                            last_epoch=param_epoch)
                if args.scheduler_type == 'ReduceLROnPlateau':
                    print("Creating ReduceLROnPlateau optimizer")
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01,
                                                  threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08, verbose=True)
                # Load Scheduler if exist
                if args.resume and checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            else:
                scheduler = None

            ##########################################
            #   _____  ____      _     ___  _   _    #
            #  |_   _||  _ \    / \   |_ _|| \ | |   #
            #    | |  | |_) |  / _ \   | | |  \| |   #
            #    | |  |  _ <  / ___ \  | | | |\  |   #
            #    |_|  |_| \_\/_/   \_\|___||_| \_|   #
            ##########################################

            ## >>>>>>>>>>>>>> ENTRY POINT FOR TRAIN <<<<<<<<<<<<<<<<<<<<<<<<

            if args.model == 'LSTM':
                train(args, feature_extractor_model, optimizer, scheduler, dataloader_train, dataloader_val,
                      os.path.basename(val_path[0]), GLOBAL_EPOCH=GLOBAL_EPOCH, LSTM=model)
            elif '360' not in args.dataloader:
                train(args, model, optimizer, scheduler, dataloader_train, dataloader_val,
                      os.path.basename(val_path[0]), GLOBAL_EPOCH=GLOBAL_EPOCH)
            else:
                train(args, model, optimizer, scheduler, dataloader_train, dataloader_val,
                      valfolder=val_sequence_list[0], GLOBAL_EPOCH=GLOBAL_EPOCH)

            if args.telegram:
                send_telegram_message("Train finished")

    ###################################
    #  _______ ______  _____ _______  #
    # |__   __|  ____|/ ____|__   __| #
    #    | |  | |__  | (___    | |    #
    #    | |  |  __|  \___ \   | |    #
    #    | |  | |____ ____) |  | |    #
    #    |_|  |______|_____/   |_|    #
    ###################################
    if args.test:
        if args.dataloader == 'Kitti360':  # Trained with RGB images or Homography
            if args.oposite:
                # Testing trained model in kitti2011 with kitti360
                test_dataset = kitti360(args.dataset, kitti360_sequence_list, transform=rgb_image_test_transforms)
            else:
                # Testing trained model with kitti360 test sequence
                test_dataset = kitti360(args.dataset, test_sequence_list, transform=rgb_image_test_transforms)

        elif args.dataloader == 'Kitti360_3D':  # Trained with 3D images
            if args.oposite:
                # Testing trained model in kitti2011 with kitti360
                test_dataset = kitti360(args.dataset, kitti360_sequence_list, transform=threedimensional_transfomrs)
            else:
                # Testing trained model with kitti360 test sequence
                test_dataset = kitti360(args.dataset, test_sequence_list, transform=threedimensional_transfomrs)

        elif args.dataloader == 'Kitti2011_RGB':  # Trained with RGB images or Homography
            if args.oposite:
                # Testing trained model in kitti360 with kitti2011
                test_dataset = Kitti2011_RGB(folders, transform=rgb_image_test_transforms)
            else:
                # Testing trained model with kitti2011 test sequence
                test_dataset = Kitti2011_RGB([test_path], transform=rgb_image_test_transforms)

        elif args.dataloader == 'generatedDataset':  # Trained with 3D images
            if args.embedding:
                if args.oposite:
                    # Testing trained model in kitti360 with kitti2011
                    test_dataset = fromGeneratedDataset(np.array(folders), args.distance,
                                                        transform=threedimensional_transfomrs)
                else:
                    # Testing trained model with kitti2011 test sequence
                    test_dataset = fromGeneratedDataset(np.array([test_path]), args.distance,
                                                        transform=threedimensional_transfomrs)

        elif args.dataloader == "triplet_OBB":
            test_dataset = triplet_OBB([test_path], args.distance, elements=200, canonical=True,
                                       transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs)
        elif args.dataloader == "triplet_BOO":
            test_dataset = triplet_BOO([test_path], args.distance, canonical=True,
                                       transform_osm=osmTransforms, transform_bev=threedimensional_transfomrs)

        elif args.dataloader == 'lstm_txt_dataloader' and not all(
                map(lambda x: '3D' in x, train_path)):  # // RGB // Homography
            print('Testing with rgb Data augmentation')
            test_dataset = lstm_txt_dataloader(test_path, transform=rgb_image_test_transforms,
                                               fixed_lenght=args.fixed_length)
        elif args.dataloader == 'lstm_txt_dataloader':  # // 3D // 3D-Masked
            print('Testing with three-dimensional data')
            test_dataset = lstm_txt_dataloader(test_path, transform=threedimensional_transfomrs,
                                               fixed_lenght=args.fixed_length)

        elif args.dataloader == 'txt_dataloader' and not all(map(lambda x: '3D' in x, train_path)):
            if args.imagenet_norm:
                print('Testing with rgb Data augmentation')
                test_dataset = txt_dataloader(test_path, transform=rgb_image_test_transforms)
            else:
                print('Testing with GAN Data augmentation')
                test_dataset = txt_dataloader(test_path, transform=GAN_transfomrs)
        elif args.dataloader == 'txt_dataloader':
            print('Testing with three-dimensional data')
            test_dataset = txt_dataloader(test_path, transform=threedimensional_transfomrs)

        # we can change the min-elements of each dataloader here
        if args.dataloader == 'lstm_txt_dataloader' and args.usesmallest:
            if 'train_dataset' in locals() and 'val_dataset' in locals():
                print("Train/Val/Test all-in-one!")
                print("Evaluating min-sequence frame value...")
                smallest = min(train_dataset.min_elements, val_dataset.min_elements)
                print("Between the TRAIN and VALIDATION data, the SMALLEST sequence has ", str(smallest), ' frames')
            else:
                # we're just testing, cant detect the smallest in train/val... so check if we have a value to use
                if args.defaultsequencelength > 0:
                    print("Using --defaultsequencelength: ", str(args.defaultsequencelength))
                    smallest = args.defaultsequencelength
                else:
                    print('Please set --defaultsequencelength, since TRAIN/VAL datasets are not provided, '
                          '(this sould be a test-only run')
                    exit(-1)
            test_dataset.min_elements = smallest

        if test_dataset.getIsSequence():
            print("Using a sequence dataset ...")
            dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=False,
                                         collate_fn=lambda x: x)
        else:
            print("Using a single-frame dataset (not a sequence) ...")
            dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, worker_init_fn=init_fn)

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.init(project=args.project, group=group_id, entity='chiringuito',
                       job_type="eval")
            wandb.config.update(args)

        # >>>>>>>>>>>> ENTRY POINT FOR TEST <<<<<<<<<<<<<<<

        if args.triplet:
            test(args, dataloader_test, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                 save_embeddings=args.save_embeddings, test_path=test_path)
        elif args.metric:
            test(args, dataloader_test, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                 test_path=test_path)
        elif args.model == 'LSTM':
            test(args, dataloader_test, test_path=test_path)
        else:
            test(args, dataloader_test, save_embeddings=args.save_embeddings)

    if args.telegram:
        send_telegram_message("Finish successfully")


if __name__ == '__main__':

    ###################################################################################
    # Workaround for r5g2 machine... opencv stuff                                     #
    # Seems related to:                                                               #
    #    1. https://github.com/opencv/opencv/issues/5150 and                          #
    #    2. https: // github.com / pytorch / pytorch / issues / 1355                  #
    # we don't know why but this is needed only in R5G2 machine (hostname NvidiaBrut) #
    ###################################################################################
    if socket.gethostname() == "NvidiaBrut":  # or "af407119309b": ~this was a test for the docker container
        print("\nDetected NvidiaBrut - Applying patch\n")
        multiprocessing.set_start_method('spawn')
    else:
        print("\nGoooood! This is not NvidiaBrut!\n")

    # basic parameters
    parser = argparse.ArgumentParser()

    ###########################################
    # SCRIPT MODALITIES AND NETWORK BEHAVIORS #
    ###########################################

    parser.add_argument('--seed', type=int, default=0, help='Starting seed, for reproducibility. Default is ZERO!')
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=False, help='Train/Validate the model')
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help='Test the model')
    parser.add_argument('--oposite', type=str2bool, nargs='?', const=True, default=False,
                        help='Test the model with the oposite dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    ### save things
    parser.add_argument('--save_embeddings', type=str2bool, nargs='?', const=True, default=False,
                        help='save embeddings')
    parser.add_argument('--saveEmbeddingsPath', type=str, default='./trainedmodels/embeddings',
                        help='path to save embbedingsf')
    parser.add_argument('--save_model_path', type=str, default='./trainedmodels/', help='path to save model')
    parser.add_argument('--save_prefix', type=str, default='', help='Prefix to all saved models')

    ### wandb stuff
    parser.add_argument('--wandb_group_id', type=str, help='Set group id for the wandb experiment')
    parser.add_argument('--project', type=str, help='wandb project name')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--wandb_resume', type=str, default=None, help='the id of the wandb-resume, e.g. jhc0gvhb')

    parser.add_argument('--telegram', type=str2bool, nargs='?', const=True, default=False,
                        help='Send info through Telegram')
    parser.add_argument('--norm_conf_matrix', type=str, default=False, choices=['index', 'columns'],
                        help='Normalization type of the confusion matix. Chooses: index, columns')
    parser.add_argument('--dataset', action="extend", nargs="+", type=str,
                        help='path to the dataset you are using. (Train or full split)')
    parser.add_argument('--dataset_val', action="extend", nargs="+", type=str, default=None,
                        help='path to the validation dataset that you are using if is different to the training one')
    parser.add_argument('--dataset_test', action="extend", nargs="+", type=str, default=None,
                        help='path to the testing dataset that you are using if is different to the training one')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each batch')
    parser.add_argument('--image_size', nargs='+', type=int, default=[224, 224], help='Number of images in each batch')
    parser.add_argument('--imagenet_norm', type=str2bool, nargs='?', const=True, default=True,
                        help='Use imagenet normalization values')
    parser.add_argument('--model', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet50 or resnet101.')
    parser.add_argument('--savemodel', type=str2bool, nargs='?', const=True, default=False,
                        help='Save pth models')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum used for train')

    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')

    parser.add_argument('--num_classes', type=int, default=7, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU is used for training')
    parser.add_argument('--use_gpu', type=str2bool, nargs='?', const=True, default=True,
                        help='whether to user gpu for training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--adam_weight_decay', type=float, default=5e-4, help='adam_weight_decay')
    parser.add_argument('--lossfunction', type=str, default='MSE',
                        choices=['MSE', 'SmoothL1', 'L1', 'focal', 'triplet', 'CrossEntropy'],
                        help='lossfunction selection')
    parser.add_argument('--metric', type=str2bool, nargs='?', const=True, default=False, help='Metric learning losses')
    parser.add_argument('--freeze', type=str2bool, nargs='?', const=True, default=False, help='freezed model + FC')
    parser.add_argument('--distance_function', type=str, default='pairwise',
                        choices=['pairwise', 'cosine', 'SNR'],
                        help='distance function selection')
    parser.add_argument('--p', type=float, default=2.0, help='p distance value')
    parser.add_argument('--test_method', type=str, default='svm', choices=['svm', 'mahalanobis', 'distance'],
                        help='testing classification method')
    parser.add_argument('--svm_mode', type=str, default='Linear', choices=['Linear', 'ovo'],
                        help='svm classification method')
    parser.add_argument('--get_scores', type=str2bool, nargs='?', const=True, default=False,
                        help='get the scores for each class')
    parser.add_argument('--patience', type=int, default=-1, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=50,
                        help='Starting epoch for patience of validation. Default, 50. ')

    parser.add_argument('--decimate', type=int, default=1, help='How much of the points will remain after '
                                                                'decimation')
    parser.add_argument('--distance', type=float, default=20.0, help='Distance from the cross')

    parser.add_argument('--weighted', type=str2bool, nargs='?', const=True, default=False, help='Weighted losses')
    parser.add_argument('--weight_tensor', type=str, default=None, choices=['Alcala', 'Kitti360', 'Kitti2011'])
    parser.add_argument('--miner', type=str2bool, nargs='?', const=True, default=False,
                        help='miner for metric learning')
    parser.add_argument('--TripletMarginMinerType', type=str, default='all', choices=['all', 'hard', 'semihard', 'easy'])
    parser.add_argument('--nonzero', type=str2bool, nargs='?', const=True, default=False, help='nonzero losses')
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use a pretrained net, or not')
    parser.add_argument('--scheduler', type=str2bool, nargs='?', const=True, default=False, help='scheduling lr')
    parser.add_argument('--scheduler_type', type=str, default='MultiStepLR', choices=['MultiStepLR',
                                                                                      'ReduceLROnPlateau'])

    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='Lstm dropout between layers')
    parser.add_argument('--fc_dropout', type=float, default=0.0, help='fc dropout between layers')
    parser.add_argument('--lstm_input', type=int, default=512, help='size of the embbedings for lstm')
    parser.add_argument('--lstm_hidden', type=int, default=256, help='size of the hidden../../DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt layers for lstm')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of layers for lstm')
    parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=False,
                        help='normalize embeddings in metric learning')

    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint model; consider check wandb_resume')

    # to enable the STUDENT training, set --embedding and provide the teacher path
    parser.add_argument('--embedding', type=str2bool, nargs='?', const=True, default=False,
                        help='Use embedding matching')
    parser.add_argument('--triplet', type=str2bool, nargs='?', const=True, default=False, help='Use embedding matching')
    parser.add_argument('--centroids_path', type=str, help='Insert centroids teacher path (for student training)')
    parser.add_argument('--label_centroids_path', type=str,
                        help='Insert label centroids teacher path (for student training)')
    parser.add_argument('--load_path', type=str, help='Insert path to the testing pth (for network testing)')
    parser.add_argument('--margin', type=float, default=1., help='margin in triplet and embedding')
    parser.add_argument('--margin_miner', type=float, default=1., help='margin in triplet and embedding')
    parser.add_argument('--feature_model', type=str, help='Feature extractor for lstm model')
    parser.add_argument('--feature_detector_path', type=str, help='Path to the feature extractor trained model')

    # different data loaders, use one from choices; a description is provided in the documentation of each dataloader
    parser.add_argument('--dataloader', type=str, default='generatedDataset',
                        choices=['fromAANETandDualBisenet', 'generatedDataset', 'Kitti2011_RGB', 'triplet_OBB',
                                 'triplet_BOO', 'triplet_ROO', 'triplet_ROO_360', 'triplet_3DOO_360', 'Kitti360',
                                 'Kitti360_3D', 'txt_dataloader', 'lstm_txt_dataloader'],
                        help='One of the supported datasets')

    parser.add_argument('--fixed_length', type=int, default=0,
                        help='fixed length strategy for lstm_txt_dataloader; '
                             '0: use all frames (no fixed length)'
                             '1: get the last min_elements, '
                             '2: linearize space and take the elements with equal-spaces')
    parser.add_argument('--defaultsequencelength', type=int, default=0, help='Set fixed length to this value')
    parser.add_argument('--usesmallest', type=str2bool, nargs='?', const=True, default=False,
                        help='Use same length for train/valid')

    parser.add_argument('--all_in_ram', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to keep images in RAM or not')

    parser.add_argument('--export_data', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to export data or not; this will be used to create .txt files, for comparison '
                             'with lstm and resnet')

    args = parser.parse_args()

    if args.get_scores and args.train:
        print("Scores are only available during test process.")
        exit(-1)

    if args.defaultsequencelength > 0 and args.fixed_length == 0:
        print(
            "Can't use defaultsequencelength > 0 for lstm with --fixed_lengh = 0 (reserved to 'use all frames' behaviour.")
        exit(-1)

    if args.oposite and not args.test:
        print("Parameter --test is REQUIRED when --oposite is set")
        exit(-1)

    if args.dataset == "":
        print("Empty path. Please provide the path of the dataset you want to use."
              "Ex: --dataset=../DualBiSeNet/data_raw")
        exit(-1)

    if args.load_path:
        if not os.path.exists(args.load_path):
            print("Load file does not exist: ", args.load_path, "\n\n")
            exit(-1)

    if args.resume:
        if not os.path.exists(args.resume):
            print("checkpoint file does not exist: ", args.resume, "\n\n")
            exit(-1)

    if args.weighted and args.nonzero:
        print("weighted and nonzero reducers selected")
        print("both reduction methods are not compatible")
        exit(-1)

    if args.weighted and args.weight_tensor is None:
        print("Weighted tensor should be selected for weighted training")
        exit(-1)

    # Ensure there's a _ at the end of the prefix
    if args.save_prefix != '':
        if args.save_prefix[-1] != '_':
            args.save_prefix = args.save_prefix + '_'

    # create a group, this is for the K-Fold https://docs.wandb.com/library/advanced/grouping#use-cases
    # K-fold cross-validation: Group together runs with different random seeds to see a larger experiment
    # group_id = wandb.util.generate_id()
    if args.wandb_group_id:
        group_id = args.wandb_group_id
    else:
        group_id = 'Kitti360_Ultimate_student'

    args.image_size = tuple(args.image_size)

    print(args)
    warnings.filterwarnings("ignore")

    if args.telegram:
        send_telegram_message(
            "Starting experiment lstm-based-intersection-classficator on " + str(socket.gethostname()))

    try:
        tic = time.time()
        main(args)
        toc = time.time()
        if args.telegram:
            send_telegram_message("Experiment of lstm-based-intersection-classficator ended after " +
                                  str(time.strftime("%H:%M:%S", time.gmtime(toc - tic))) + "\n" + "Run was on: " +
                                  str(socket.gethostname()))

    except KeyboardInterrupt:
        print("Shutdown requested")
        if args.telegram:
            send_telegram_message("Shutdown requested on " + str(socket.gethostname()))
    except Exception as e:
        if isinstance(e, SystemExit):
            exit()
        print(e)
        if args.telegram:
            send_telegram_message(
                "Error catched in lstm-based-intersection-classficator :" + str(e) + "\nRun was on: " +
                str(socket.gethostname()))
