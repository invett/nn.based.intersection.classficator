import argparse
import os
import numpy as np
import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloaders.sequencedataloader import SequenceDataset
from model.resnet_models import get_model_resnet, get_model_resnext
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def validation(args, model, criterion, dataloader_val):
    model.eval()
    loss_record = 0.0
    conf_matrix = np.zeros((args.num_classes, args.num_classes))
    for sample in dataloader_val:
        data = sample['image']
        label = sample['label']

        if torch.cuda.is_available() and args.use_gpu:
            data = data.cuda()
            label = label.cuda()

        output = model(data)

        loss = criterion(output, label)
        loss_record += loss.item()

        predict = torch.argmax(output, 1)
        conf_matrix += confusion_matrix(label, predict)

    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for validation : %f' % loss_val_mean)
    acc, acc_perclass, precision_perclass, recall_perclass = computemetrics(conf_matrix) ## Nope
    return acc, acc_perclass, precision_perclass, recall_perclass


def train(args, model, optimizer, dataloader_train, dataloader_val, acc_pre):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    kfold_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        model.zero_grad()

        for sample in dataloader_train:
            data = sample['image']
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

        tq.close()
        loss_train_mean = loss_record / len(dataloader_train)
        print('loss for train : %f' % loss_train_mean)

        if epoch % args.validation_step == 0:
            acc = validation(args, model, criterion, dataloader_val)

            if kfold_acc < acc:
                patience = 0
                kfold_acc = acc
                if acc_pre < kfold_acc:
                    bestModel = model.state_dict()
                    acc_pre = kfold_acc
                    print('Best global accuracy\n')
                    print('Saving model: ', os.path.join(args.save_model_path, 'model_{}.pth'.format(args.model)))
                    torch.save(bestModel, os.path.join(args.save_model_path, 'model_{}.pth'.format(args.model)))
            else:
                patience += 1

        if patience >= args.patience:
            break

    with open('acc_log.txt', 'a') as logfile:
        logfile.write('Acc: {}\n'.format(acc_pre))
        print('acc: {}\n'.format(acc_pre))

    return acc_pre


def main(args):
    # Acuracy acumulator
    acc = 0.0

    # create dataset and dataloader
    data_path = args.data

    dataset = SequenceDataset(data_path,
                              transform=transforms.Compose([
                                  transforms.Resize(224, 224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])
                              ]))

    kf = KFold(n_splits=10, shuffle=False)

    for i, train_index, test_index in enumerate(kf.split(list(range(len(dataset))))):
        train_data = torch.utils.data.Subset(dataset, train_index)
        test_data = torch.utils.data.Subset(dataset, test_index)

        dataloader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        dataloader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Build model
        if args.model[0:6] == 'resnet':
            model = get_model_resnet(args.model)
        elif args.model[0:7] == 'resnext':
            model = get_model_resnext(args.model)
        else:
            print('not supported model \n')
            exit()

            # Build optimizer
            # build optimizer
            if args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, momentum=args.momentum)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
            elif args.optimizer == 'ASGD':
                optimizer = torch.optim.ASGD(model.parameters(), args.learning_rate)
            elif args.optimizer == 'Adamax':
                optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate)
            else:
                print('not supported optimizer \n')
                return None

        # train model
        acc = train(args, model, optimizer, dataloader_train, dataloader_test, acc)


if __name__ == '__main__':
    pass
