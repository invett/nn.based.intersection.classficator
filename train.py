import argparse
import os
import numpy as np
import tqdm
import pandas as pd

import warnings

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev, Mirror
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.sequencedataloader import TestDataset, fromAANETandDualBisenet, BaseLine
from model.resnet_models import get_model_resnet, get_model_resnext
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt

import wandb
import seaborn as sn

import sys

from dl_bot import DLBot

telegram = False

if telegram:
    telegram_token = "1178257144:AAH5DEYxJjPb0Qm_afbGTuJZ0-oqfIMFlmY"  # replace TOKEN with your bot's token
    telegram_user_id = None  # replace None with your telegram user id (integer):
    # Create a DLBot instance
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    # Activate the bot
    bot.activate_bot()


def test(args, dataloader_test):
    print('start Test!')

    criterion = torch.nn.CrossEntropyLoss()

    # Build model
    if args.resnetmodel[0:6] == 'resnet':
        model = get_model_resnet(args.resnetmodel, args.num_classes)
    elif args.resnetmodel[0:7] == 'resnext':
        model = get_model_resnext(args.resnetmodel, args.num_classes)
    else:
        print('not supported model \n')
        exit()

    # load Saved Model
    savepath = './trainedmodels/model_' + args.resnetmodel + '.pth'
    print('load model from {} ...'.format(savepath))
    model.load_state_dict(torch.load(savepath))
    print('Done!')
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    report, confusion_matrix, acc, _ = validation(args, model, criterion, dataloader_test)

    labels_all = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in labels_all], columns=[i for i in labels_all])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    wandb.log({"Test/Acc": acc,
               "conf-matrix_test": wandb.Image(plt)})


def validation(args, model, criterion, dataloader_val):
    print('\nstart val!')
    model.eval()
    loss_record = 0.0
    labellist = np.array([])
    predlist = np.array([])

    for sample in dataloader_val:
        data = sample['data']
        label = sample['label']

        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        output = model(data)

        loss = criterion(output, label)
        loss_record += loss.item()

        predict = torch.argmax(output, 1)
        label = label.squeeze().cpu().numpy()
        predict = predict.squeeze().cpu().numpy()

        labellist = np.append(labellist, label)
        predlist = np.append(predlist, predict)

    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for test/validation : %f' % loss_val_mean)

    # Calculate validation metrics
    conf_matrix = confusion_matrix(labellist, predlist, labels=[0, 1, 2, 3, 4, 5, 6])
    report_dict = classification_report(labellist, predlist, output_dict=True, zero_division=0)
    acc = accuracy_score(labellist, predlist)
    print('Accuracy for test/validation : %f\n' % acc)

    return report_dict, conf_matrix, acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, acc_pre, valfolder):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    kfold_acc = 0.0
    kfold_loss = np.inf
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', cooldown=2, patience=2)

    wandb.watch(model, log="all")

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        trainlabellist = np.array([])
        trainpredlist = np.array([])

        for sample in dataloader_train:
            data = sample['data']
            label = sample['label']

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            output = model(data)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()
            predict = torch.argmax(output, 1)
            label = label.squeeze().cpu().numpy()
            predict = predict.squeeze().cpu().numpy()

            trainlabellist = np.append(trainlabellist, label)
            trainpredlist = np.append(trainpredlist, predict)

        tq.close()
        loss_train_mean = loss_record / len(dataloader_train)
        acc_train = accuracy_score(trainlabellist, trainpredlist)
        print('loss for train : %f' % loss_train_mean)
        print('acc for train : %f' % acc_train)

        wandb.log({"Train/loss": loss_train_mean,
                   "Train/acc": acc_train,
                   "Train/lr": lr})

        if epoch % args.validation_step == 0:
            report, confusion_matrix, acc_val, loss_val = validation(args, model, criterion, dataloader_val)
            scheduler.step(loss_val)

            labels_all = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
            df_cm = pd.DataFrame(confusion_matrix, index=[i for i in labels_all], columns=[i for i in labels_all])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)

            wandb.log({"Val/loss": loss_val,
                       "Val/Acc": acc_val,
                       "conf-matrix_{}_{}".format(valfolder, epoch): wandb.Image(plt)})

            if kfold_acc < acc_val or kfold_loss > loss_train_mean:
                patience = 0
                if kfold_acc < acc_val:
                    kfold_acc = acc_val
                else:
                    kfold_loss = loss_train_mean
                if acc_pre < kfold_acc:
                    bestModel = model.state_dict()
                    acc_pre = kfold_acc
                    print('Best global accuracy: {}'.format(kfold_acc))
                    print('Saving model: ', os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                    torch.save(bestModel, os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                    wandb.save(os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))

            elif epoch < args.patience_start:
                patience = 0

            else:
                patience += 1

        if patience >= args.patience > 0:
            break

    return acc_pre


def main(args, model=None):
    # Acuracy acumulator
    acc = 0.0

    # create dataset and dataloader
    data_path = args.dataset

    # All sequence folders
    folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                        os.path.isdir(os.path.join(data_path, folder))])

    # Exclude test samples
    folders = folders[folders != os.path.join(data_path, '2011_10_03_drive_0027_sync')]
    test_path = os.path.join(data_path, '2011_10_03_drive_0027_sync')

    try:
        loo = LeaveOneOut()
        for train_index, val_index in loo.split(folders):
            train_path, val_path = folders[train_index], folders[val_index]
            if args.bev:
                val_dataset = fromAANETandDualBisenet(val_path, transform=transforms.Compose([Normalize(),
                                                                                              GenerateBev(
                                                                                                  decimate=args.decimate),
                                                                                              Mirror(),
                                                                                              Rescale((224, 224)),
                                                                                              ToTensor()]))

                train_dataset = fromAANETandDualBisenet(train_path, transform=transforms.Compose([Normalize(),
                                                                                                  GenerateBev(
                                                                                                      decimate=args.decimate),
                                                                                                  Mirror(),
                                                                                                  Rescale((224, 224)),
                                                                                                  ToTensor()]))
            else:
                val_dataset = BaseLine(val_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize(
                                                                                   (0.485, 0.456, 0.406),
                                                                                   (0.229, 0.224, 0.225))
                                                                               ]))
                train_dataset = BaseLine(train_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                   transforms.RandomAffine(15,translate=(0.0,0.1),shear=(-15,15)),
                                                                                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                                                                   ransforms.RandomPerspective(),
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225))
                                                                                   ]))

            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers)
            dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers)

            # Build model
            if args.resnetmodel[0:6] == 'resnet':
                model = get_model_resnet(args.resnetmodel, args.num_classes)
            elif args.resnetmodel[0:7] == 'resnext':
                model = get_model_resnext(args.resnetmodel, args.num_classes)
            else:
                print('not supported model \n')
                exit()
            if torch.cuda.is_available() and args.use_gpu:
                model = model.cuda()

            # build optimizer
            if args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=args.momentum)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.lr)
            elif args.optimizer == 'ASGD':
                optimizer = torch.optim.ASGD(model.parameters(), args.lr)
            elif args.optimizer == 'Adamax':
                optimizer = torch.optim.Adamax(model.parameters(), args.lr)
            else:
                print('not supported optimizer \n')
                exit()

            # train model
            acc = train(args, model, optimizer, dataloader_train, dataloader_val, acc, os.path.basename(val_path[0]))

            if telegram:
                bot.send_message("K-Fold finished")

    except:  # catch *all* exceptions
        e = sys.exc_info()
        print(e)

        if telegram:
            bot.send_message(str(e))

        exit()

    # Final Test on 2011_10_03_drive_0027_sync
    if args.bev:
        test_dataset = TestDataset(test_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                                                 (0.229, 0.224, 0.225))
                                                                            ]))
    else:
        test_dataset = BaseLine([test_path], transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                                                (0.229, 0.224, 0.225))
                                                                           ]))

    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    test(args, dataloader_test)

    if telegram:
        bot.send_message("Finish successfully")


if __name__ == '__main__':
    # basic parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--dataset', type=str, help='path to the dataset you are using.')
    parser.add_argument('--bev', action='store_true', help='path to the dataset you are using.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
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

    parser.add_argument('--decimate', type=float, default=0.2, help='How much of the points will remain after '
                                                                    'decimation')

    args = parser.parse_args()

    # create a group, this is for the K-Fold https://docs.wandb.com/library/advanced/grouping#use-cases
    # K-fold cross-validation: Group together runs with different random seeds to see a larger experiment
    group_id = wandb.util.generate_id()
    wandb.init(project="nn-based-intersection-classficator", group=group_id, job_type="training")
    wandb.config.update(args)
    print(args)
    warnings.filterwarnings("ignore")

    if telegram:
        bot.send_message('Starting experiment nn-based-intersection-classficator')
    main(args)
