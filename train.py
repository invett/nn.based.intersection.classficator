import argparse
import os
import numpy as np
import tqdm
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.sequencedataloader import SequenceDataset, fromAANETandDualBisnet
from model.resnet_models import get_model_resnet, get_model_resnext
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorboardX import SummaryWriter


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
    print('loss for validation : %f' % loss_val_mean)

    # Calculate validation metrics
    conf_matrix = confusion_matrix(labellist, predlist)
    report_dict = classification_report(labellist, predlist, output_dict=True, zero_division=0)
    acc = accuracy_score(labellist, predlist)
    print('Accuracy for validation : %f\n' % acc)

    return report_dict, conf_matrix, acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, acc_pre):
    writer = SummaryWriter()
    step = 0

    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    kfold_acc = 0.0
    kfold_loss = np.inf
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        model.zero_grad()

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
            model.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()
            writer.add_scalar('Train/loss_step', loss.item(), step)
            step += 1

        tq.close()
        loss_train_mean = loss_record / len(dataloader_train)
        writer.add_scalar('Train/loss_epoch', loss_train_mean, epoch)
        print('loss for train : %f' % loss_train_mean)

        if epoch % args.validation_step == 0:
            report, confusion_matrix, acc, val_loss = validation(args, model, criterion, dataloader_val)
            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/acc', acc, epoch)
            if kfold_acc < acc or kfold_loss > loss_train_mean:
                patience = 0
                kfold_acc = acc
                kfold_loss = loss_train_mean
                if acc_pre < kfold_acc:
                    bestModel = model.state_dict()
                    acc_pre = kfold_acc
                    print('Best global accuracy: {}'.format(kfold_acc))
                    print('Saving model: ', os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                    torch.save(bestModel, os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                    print('Saving report and confusion matrix\n')
                    df_report = pd.DataFrame.from_dict(report)
                    df_report = df_report.transpose()
                    df_report.to_csv('report.csv', sep=';')
                    df_matrix = pd.DataFrame(data=confusion_matrix)
                    df_matrix.to_csv('confusionMatrix.csv', sep=';')

            elif epoch > args.patience_start:
                patience = 0

            else:
                patience += 1

        if patience >= args.patience > 0:
            break

    writer.close()

    with open('acc_log.txt', 'a') as logfile:
        logfile.write('Acc: {}\n'.format(kfold_acc))
        print('acc: {}\n'.format(kfold_acc))

    return acc_pre


def main(args):
    # Acuracy acumulator
    acc = 0.0

    # create dataset and dataloader
    data_path = args.dataset

    if args.dataloader == "SequenceDataset":
        dataset = SequenceDataset(data_path,
                                  transform=transforms.Compose([Rescale((224, 224)),
                                                                Normalize(),
                                                                ToTensor()
                                                                ]))
    elif args.dataloader == "fromAANETandDualBisnet":
        dataset = fromAANETandDualBisnet(data_path,
                                         transform=transforms.Compose([GenerateBev(decimate=0.2),
                                                                       Rescale((224, 224)),
                                                                       Normalize(),
                                                                       ToTensor()]))

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(list(range(len(dataset)))):
        train_data_sampler = SubsetRandomSampler(train_index)
        test_data_sampler = SubsetRandomSampler(test_index)

        print('Train data size: {}'.format(len(train_index)))
        print('Test data size: {}\n'.format(len(test_index)))

        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_data_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers)
        dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_data_sampler, shuffle=False,
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
        acc = train(args, model, optimizer, dataloader_train, dataloader_test, acc)


if __name__ == '__main__':
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', type=str, default="SequenceDataset",
                        help='Dataloader to use (SequenceDataset, fromAANETandDualBisnet)')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--dataset', type=str, help='path to the dataset you are using.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
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
    args = parser.parse_args()

    print(args)
    main(args)
