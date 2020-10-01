import argparse
import copy
import multiprocessing
import os
import socket  # to get the machine name
import time
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.transforms as transforms
import tqdm
import wandb
from sklearn.model_selection import LeaveOneOut
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataloaders.sequencedataloader import BaseLine, TestDataset, fromAANETandDualBisenet, fromGeneratedDataset, \
    triplet_BOO, triplet_OBB
from dataloaders.transforms import GenerateBev, GrayScale, Mirror, Normalize, Rescale, ToTensor
from dropout_models import get_resnet, get_resnext
from miscellaneous.utils import init_function, reset_wandb_env, send_telegram_message, send_telegram_picture, \
    student_network_pass
from model.resnet_models import Personalized, Personalized_small, get_model_resnet, get_model_resnext


def test(args, dataloader_test, gt_model=None):
    print('start Test!')

    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin)
    elif args.embedding:
        criterion = torch.nn.CosineEmbeddingLoss(margin=args.margin)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Build model
    if args.resnetmodel[0:6] == 'resnet':
        model = get_model_resnet(args.resnetmodel, args.num_classes, transfer=args.transfer, pretrained=args.pretrained,
                                 embedding=(args.embedding or args.triplet))
    elif args.resnetmodel[0:7] == 'resnext':
        model = get_model_resnext(args.resnetmodel, args.num_classes, args.transfer, args.pretrained)
    elif args.resnetmodel == 'personalized':
        model = Personalized(args.num_classes)
    else:
        model = Personalized_small(args.num_classes)

    # load Saved Model
    savepath = './trainedmodels/model_' + args.resnetmodel + '.pth'
    print('load model from {} ...'.format(savepath))
    model.load_state_dict(torch.load(savepath))
    print('Done!')

    if args.embedding:
        gt_model = copy.deepcopy(model)
        gt_model.load_state_dict(torch.load(args.teacher_path))
        gt_model.eval()

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()
        if args.embedding:
            gt_model = gt_model.cuda()

    # Start testing
    if args.embedding:
        acc_val, loss_val = validation(args, model, criterion, dataloader_test, gtmodel=gt_model)
        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Test/loss": loss_val, "Test/Acc": acc_val})
    elif args.triplet:
        pass
    else:
        confusion_matrix, acc, _ = validation(args, model, criterion, dataloader_test)

        plt.figure(figsize=(10, 7))
        sn.heatmap(confusion_matrix, annot=True, fmt='.3f')

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Test/Acc": acc, "conf-matrix_test": wandb.Image(plt)})


def validation(args, model, criterion, dataloader_val, gtmodel=None):
    print('\nstart val!')

    loss_record = 0.0
    acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    with torch.no_grad():
        model.eval()

        for sample in dataloader_val:
            if args.embedding or args.triplet:
                acc, loss = student_network_pass(args, sample, criterion, model, gtmodel)
            else:
                acc, loss, label, predict = student_network_pass(args, sample, criterion, model)
                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)

            loss_record += loss.item()
            acc_record += acc

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for test/validation : %f' % loss_val_mean)

    if args.triplet or args.embedding:
        acc = acc_record / (len(dataloader_val) * args.batch_size)
    else:
        acc = acc_record / len(dataloader_val)
    print('Accuracy for test/validation : %f\n' % acc)

    if not (args.embedding or args.triplet):
        conf_matrix = pd.crosstab(labelRecord, predRecord, rownames=['Actual'], colnames=['Predicted'], margins=True,
                                  normalize='all')
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6, 'All'], columns=[0, 1, 2, 3, 4, 5, 6, 'All'],
                                          fill_value=0)
        return conf_matrix, acc, loss_val_mean
    else:
        return acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, acc_pre, valfolder, gtmodel=None):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    kfold_acc = 0.0
    kfold_loss = np.inf

    # Build loss criterion
    if args.weighted:
        weights = [0.91, 0.95, 0.96, 0.84, 0.85, 0.82, 0.67]
        class_weights = torch.FloatTensor(weights).cuda()
        traincriterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        valcriterion = torch.nn.CrossEntropyLoss()
    elif args.embedding:
        traincriterion = torch.nn.CosineEmbeddingLoss(margin=args.margin, reduction='mean')
        valcriterion = torch.nn.CosineEmbeddingLoss(margin=args.margin, reduction='mean')
    elif args.triplet:
        traincriterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2.0, reduction='mean')
        valcriterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2.0, reduction='mean')
    else:
        traincriterion = torch.nn.CrossEntropyLoss()
        valcriterion = torch.nn.CrossEntropyLoss()

    model.zero_grad()
    model.train()
    scheduler = MultiStepLR(optimizer, milestones=[10, 40, 80], gamma=0.5)

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.watch(model, log="all")

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        acc_record = 0.0

        for sample in dataloader_train:
            if args.embedding or args.triplet:
                acc, loss = student_network_pass(args, sample, traincriterion, model, gtmodel)
            else:
                acc, loss, _, _ = student_network_pass(args, sample, traincriterion, model)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()
            acc_record += acc

        tq.close()

        if args.scheduler:
            scheduler.step()

        # Calculate metrics
        loss_train_mean = loss_record / len(dataloader_train)
        print('loss for train : %f' % loss_train_mean)

        if args.triplet or args.embedding:
            acc_train = acc_record / (len(dataloader_train) * args.batch_size)
        else:
            acc_train = acc_record / len(dataloader_train)
        print('acc for train : %f' % acc_train)

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Train/loss": loss_train_mean,
                       "Train/acc": acc_train,
                       "Train/lr": lr}, step=epoch)

        if epoch % args.validation_step == 0:
            if args.embedding or args.triplet:
                acc_val, loss_val = validation(args, model, valcriterion, dataloader_val, gtmodel=gtmodel)
                if not args.nowandb:  # if nowandb flag was set, skip
                    wandb.log({"Val/loss": loss_val, "Val/Acc": acc_val}, step=epoch)
            else:
                confusion_matrix, acc_val, loss_val = validation(args, model, valcriterion, dataloader_val)
                plt.figure(figsize=(10, 7))
                sn.heatmap(confusion_matrix, annot=True, fmt='.3f')

                if args.telegram:
                    send_telegram_picture(plt, "Epoch:" + str(epoch))

                if not args.nowandb:  # if nowandb flag was set, skip
                    wandb.log({"Val/loss": loss_val,
                               "Val/Acc": acc_val,
                               "conf-matrix_{}_{}".format(valfolder, epoch): wandb.Image(plt)}, step=epoch)

            if (kfold_acc < acc_val) or (kfold_loss > loss_val):

                patience = 0
                print('Patience restart')

                if kfold_acc < acc_val:
                    kfold_acc = acc_val
                if kfold_loss > loss_val:
                    kfold_loss = loss_val

                if acc_pre < kfold_acc:
                    bestModel = model.state_dict()
                    acc_pre = kfold_acc
                    print('Best global accuracy: {}'.format(kfold_acc))
                    print('Saving model: ', os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                    torch.save(bestModel, os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))

                    if not args.nowandb:  # if nowandb flag was set, skip
                        wandb.save(os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))

            elif epoch < args.patience_start:
                patience = 0
                print('Patience start not reached')

            else:
                patience += 1
                print('Patience: {}\n'.format(patience))

        if patience >= args.patience > 0:
            break

    return acc_pre


def main(args, model=None):
    # Try to avoid randomness -- https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    hyperparameter_defaults = dict(batch_size=64, cuda='0', dataloader='BaseLine', dataset=None, decimate=1.0,
                                   distance=20.0, embedding=False, embedding_class=False, grayscale=False, lr=0.0001,
                                   margin=0.5, momentum=0.9, nowandb=False, num_classes=7, num_epochs=50, num_workers=4,
                                   optimizer='sgd', patience=-1, patience_start=50, pretrained=False,
                                   resnetmodel='resnet18', save_model_path='./trainedmodels/', scheduler=False, seed=0,
                                   sweep=False, teacher_path=None, telegram=False, test=False, transfer=False,
                                   triplet=False, use_gpu=True, validation_step=5, weighted=False)

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = "URL" #sweep_run.get_sweep_url()
    project_url = "PROJECTURL"  # sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    gigi = wandb.config
    print("This is the config:")
    print(gigi)
    print("====================================================================================")
    print(sweep_id)         this is the 'group'
    print(sweep_run_name)   this will be the base name for all the K-folds, just add some number after, like the K-fold
                            fold so to have --->>>  still-sweep-7-0
                                                    still-sweep-7-1
                                                    still-sweep-7-2 all these shares the same configuration from the sweep
                                                    still-sweep-7-3
                                                    still-sweep-7-4
                            where still-sweep-7 is the sweep name.
    print("====================================================================================")

    if args.sweep:
        print("YES IT IS A SWEEP")
    else:
        print("VERY SAD TIMES....")
    exit(-1)

    # Getting the hostname to add to wandb (seem useful for sweeps)
    hostname = str(socket.gethostname())

    GLOBAL_EPOCH = multiprocessing.Value('i', 0)
    seed = multiprocessing.Value('i', args.seed)

    init_fn = partial(init_function, seed=seed, epoch=GLOBAL_EPOCH)

    # Accuracy accumulator
    acc = 0.0

    # create dataset and dataloader
    data_path = args.dataset

    if data_path == "":
        print("Empty path. Please provide the path of the dataset you want to use."
              "Ex: --dataset=../DualBiSeNet/data_raw")
        exit(-1)

    # All sequence folders
    folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                        os.path.isdir(os.path.join(data_path, folder))])

    # Exclude test samples
    folders = folders[folders != os.path.join(data_path, '2011_09_30_drive_0028_sync')]
    test_path = os.path.join(data_path, '2011_09_30_drive_0028_sync')

    if args.grayscale:
        aanetTransforms = transforms.Compose(
            [GenerateBev(decimate=args.decimate), Mirror(), Rescale((224, 224)), Normalize(), GrayScale(), ToTensor()])
        generateTransforms = transforms.Compose([Rescale((224, 224)), Normalize(), GrayScale(), ToTensor()])
    else:
        aanetTransforms = transforms.Compose(
            [GenerateBev(decimate=args.decimate), Mirror(), Rescale((224, 224)), Normalize(), ToTensor()])
        generateTransforms = transforms.Compose([Rescale((224, 224)), Normalize(), ToTensor()])
        obsTransforms = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

    if not args.test:
        loo = LeaveOneOut()
        for train_index, val_index in loo.split(folders):

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.init(project="nn-based-intersection-classficator", group=group_id, entity='chiringuito',
                           job_type="training", reinit=True)
                wandb.config.update(args)

            train_path, val_path = folders[train_index], folders[val_index]

            if args.dataloader == "fromAANETandDualBisenet":
                val_dataset = fromAANETandDualBisenet(val_path, args.distance, transform=aanetTransforms)
                train_dataset = fromAANETandDualBisenet(train_path, args.distance, transform=aanetTransforms)

            elif args.dataloader == "generatedDataset":
                val_dataset = fromGeneratedDataset(val_path, args.distance, transform=generateTransforms)
                train_dataset = fromGeneratedDataset(train_path, args.distance, transform=generateTransforms)

            elif args.dataloader == "triplet_OBB":

                val_dataset = triplet_OBB(val_path, args.distance, elements=200, canonical=False,

                                          transform_obs=obsTransforms, transform_bev=generateTransforms)

                train_dataset = triplet_OBB(train_path, args.distance, elements=2000, canonical=False,

                                            transform_obs=obsTransforms, transform_bev=generateTransforms)

            elif args.dataloader == "triplet_BOO":

                val_dataset = triplet_BOO(val_path, args.distance, elements=200, canonical=False,

                                          transform_obs=obsTransforms, transform_bev=generateTransforms)

                train_dataset = triplet_BOO(train_path, args.distance, elements=2000, canonical=False,

                                            transform_obs=obsTransforms, transform_bev=generateTransforms)

            elif args.dataloader == "BaseLine":
                val_dataset = BaseLine(val_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize(
                                                                                   (0.485, 0.456, 0.406),
                                                                                   (0.229, 0.224, 0.225))
                                                                               ]))
                train_dataset = BaseLine(train_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                   transforms.RandomAffine(15,
                                                                                                           translate=(
                                                                                                               0.0,
                                                                                                               0.1),
                                                                                                           shear=(
                                                                                                               -15,
                                                                                                               15)),
                                                                                   transforms.ColorJitter(
                                                                                       brightness=0.5, contrast=0.5,
                                                                                       saturation=0.5),
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225))
                                                                                   ]))
            else:
                raise Exception("Dataloader not found")

            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, worker_init_fn=init_fn)
            dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, worker_init_fn=init_fn)

            # Build model
            if args.resnetmodel[0:6] == 'resnet':
                model = get_model_resnet(args.resnetmodel, args.num_classes, transfer=args.transfer,
                                         pretrained=args.pretrained,
                                         embedding=(args.embedding or args.triplet) and not args.embedding_class)
            elif args.resnetmodel[0:7] == 'resnext':
                model = get_model_resnext(args.resnetmodel, args.num_classes, args.transfer, args.pretrained)
            elif args.resnetmodel == 'personalized':
                model = Personalized(args.num_classes)
            elif args.resnetmodel == 'personalized_small':
                model = Personalized_small(args.num_classes)
            elif args.dropout:
                if args.resnetmodel == 'resnet':
                    model = get_resnext(args, args.cardinality, args.d_width, args.num_classes)
                else:
                    model = get_resnet(args, args.cardinality)

            if args.embedding:
                gt_model = copy.deepcopy(model)
                gt_model.load_state_dict(torch.load(args.teacher_path))
                if args.embedding_class:  # if I'm using the teacher trained with FC I need to get rid of it before.
                    model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    gt_model = torch.nn.Sequential(*(list(gt_model.children())[:-1]))
                gt_model.eval()

            if torch.cuda.is_available() and args.use_gpu:
                model = model.cuda()
                if args.embedding:
                    gt_model = gt_model.cuda()

            # build optimizer
            if args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=args.momentum)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
            elif args.optimizer == 'ASGD':
                optimizer = torch.optim.ASGD(model.parameters(), args.lr)
            elif args.optimizer == 'Adamax':
                optimizer = torch.optim.Adamax(model.parameters(), args.lr)
            else:
                print('not supported optimizer \n')
                exit()

            # train model
            if args.embedding:
                acc = train(args, model, optimizer, dataloader_train, dataloader_val, acc,
                            os.path.basename(val_path[0]), gtmodel=gt_model)
            else:
                acc = train(args, model, optimizer, dataloader_train, dataloader_val, acc,
                            os.path.basename(val_path[0]))

            if args.telegram:
                send_telegram_message("K-Fold finished")

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.join()

    # Final Test on 2011_10_03_drive_0027_sync
    if args.dataloader == "fromAANETandDualBisenet":
        test_dataset = TestDataset(test_path, args.distance,
                                   transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                                                      (0.229, 0.224, 0.225))
                                                                 ]))
    elif args.dataloader == 'BaseLine':
        test_dataset = BaseLine([test_path], transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                                                (0.229, 0.224, 0.225))
                                                                           ]))
    elif args.dataloader == 'generatedDataset':
        if args.embedding:
            test_dataset = fromGeneratedDataset([test_path], args.distance, transform=generateTransforms)
        else:
            test_path = test_path.replace('data_raw_bev', 'data_raw')
            test_dataset = TestDataset(test_path, args.distance, transform=generateTransforms)

    elif args.dataloader == "triplet_OBB":
        test_dataset = triplet_OBB([test_path], args.distance, elements=200, canonical=True,
                                   transform_obs=obsTransforms, transform_bev=generateTransforms)
    elif args.dataloader == "triplet_BOO":
        test_dataset = triplet_BOO([test_path], args.distance, elements=200, canonical=True,
                                   transform_obs=obsTransforms, transform_bev=generateTransforms)

    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 worker_init_fn=init_fn)

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.init(project="nn-based-intersection-classficator", group=group_id, entity='chiringuito', job_type="eval")
        wandb.config.update(args)

    test(args, dataloader_test)

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
    if socket.gethostname() == "NvidiaBrut":
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
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--sweep', action='store_true', help='if set, this run is part of a wandb-sweep; use it with'
                                                             'as documented in '
                                                             'in https://docs.wandb.com/sweeps/configuration#command')

    parser.add_argument('--telegram', action='store_true', help='Send info through Telegram')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--dataset', type=str, help='path to the dataset you are using.')
    parser.add_argument('--transfer', action='store_true', help='Fine tuning or transfer learning')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each batch')
    parser.add_argument('--resnetmodel', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet50 or resnet101.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum used for train')

    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=7, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU is used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--save_model_path', type=str, default='./trainedmodels/', help='path to save model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--patience', type=int, default=-1, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=50,
                        help='Starting epoch for patience of validation. Default, 50. ')

    parser.add_argument('--decimate', type=float, default=1.0, help='How much of the points will remain after '
                                                                    'decimation')
    parser.add_argument('--distance', type=float, default=20.0, help='Distance from the cross')

    parser.add_argument('--weighted', action='store_true', help='Weighted losses')
######    parser.add_argument('--pretrained', action='store_true', help='pretrained net')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use a pretrained net, or not')
#####    parser.add_argument('--scheduler', action='store_true', help='scheduling lr')
    parser.add_argument('--scheduler', type=bool, default=False, help='scheduling lr')
    parser.add_argument('--test', action='store_true', help='scheduling lr')
    parser.add_argument('--grayscale', action='store_true', help='Use Grayscale Images')

    # to enable the STUDENT training, set --embedding and provide the teacher path
    parser.add_argument('--embedding', action='store_true', help='Use embedding matching')
    parser.add_argument('--embedding_class', action='store_true', help='Use embedding matching with classification')
    parser.add_argument('--triplet', action='store_true', help='Use triplet learing')
    parser.add_argument('--teacher_path', type=str, help='Insert teacher path (for student training)')
    parser.add_argument('--margin', type=float, default=0.5, help='margin in triplet and embedding')

    # different data loaders, use one from choices; a description is provided in the documentation of each dataloader
    parser.add_argument('--dataloader', type=str, default='BaseLine', choices=['fromAANETandDualBisenet',
                                                                               'generatedDataset',
                                                                               'BaseLine',
                                                                               'triplet_OBB',
                                                                               'TestDataset'],
                        help='One of the supported datasets')

    subparsers = parser.add_subparsers(help='Subparser for dropout models')
    parser_drop = subparsers.add_parser('dropout', help='Dropout models. Resnext and Resnet')
    parser_drop.add_argument('--cardinality', type=int, default=10, help='addition, e.g. widen_factor')
    parser_drop.add_argument('--d_with', type=int, default=64, help='addition arg2, e.g. base_width (ResNeXt)')
    parser_drop.add_argument('--depth', type=int, default=100, help='depth of network')
    parser_drop.add_argument('--block_type', type=int, default=1, help='specify block_type (default: 1)')
    parser_drop.add_argument('--use_gn', action='store_true', default=False,
                             help='whether to use group norm (default: False)')
    parser_drop.add_argument('--gn_groups', type=int, default=8, help='group norm groups')
    parser_drop.add_argument('--drop_type', type=int, default=0,
                             help='0-drop-neuron, 1-drop-channel, 2-drop-path, 3-drop-layer')
    parser_drop.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate')
    parser_drop.add_argument('--report_ratio ', action='store_true', help='Cardinality of the net')

    args = parser.parse_args()

    # check whether --embedding was set but with no teacher path
    if args.embedding and not args.teacher_path:
        print("Parameter --teacher_path is REQUIRED when --embedding is set")
        exit(-1)

    # create a group, this is for the K-Fold https://docs.wandb.com/library/advanced/grouping#use-cases
    # K-fold cross-validation: Group together runs with different random seeds to see a larger experiment
    # group_id = wandb.util.generate_id()
    group_id = 'Teacher_Student_embedding'
    print(args)
    warnings.filterwarnings("ignore")

    if args.telegram:
        send_telegram_message("Starting experiment nn-based-intersection-classficator")

    try:
        tic = time.time()
        main(args)
        toc = time.time()
        if args.telegram:
            send_telegram_message("Experiment of nn-based-intersection-classficator ended after " +
                                  str(time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

    except KeyboardInterrupt:
        print("Shutdown requested")
        if args.telegram:
            send_telegram_message("Shutdown requested")
    except Exception as e:
        if isinstance(e, SystemExit):
            exit()
        print(e)
        if args.telegram:
            send_telegram_message("Error catched in nn-based-intersection-classficator :" + str(e))
