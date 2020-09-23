import argparse
import os
import numpy as np
import tqdm
import pandas as pd

import torchvision.transforms as transforms
from dataloaders.sequencedataloader import teacher_tripletloss_generated, teacher_tripletloss

import warnings

import torch
from torch.utils.data import DataLoader
from torch import nn

from model.resnet_models import get_model_resnet, get_model_resnext, Personalized, Personalized_small
from dropout_models import get_resnext, get_resnet
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from miscellaneous.utils import send_telegram_message

import matplotlib.pyplot as plt

import wandb
import seaborn as sn


def main(args):
    addnoise = True
    if args.no_noise:
        addnoise = False

    if not args.nowandb:  # if nowandb flag was set, skip
        if args.test:
            wandb.init(project="nn-based-intersection-classficator", entity="chiringuito", group="Teacher_train",
                       job_type="eval")
        else:
            wandb.init(project="nn-based-intersection-classficator", entity="chiringuito", group="Teacher_train",
                       job_type="training")
        wandb.config.update(args)

    # Build Model
    model = get_model_resnet(args.resnetmodel, args.num_classes, greyscale=False, embedding=args.triplet)

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

    if args.train:
        # In both, set canonical to False to speedup the process; canonical won't be used for train/validate.
        dataset_train = teacher_tripletloss_generated(elements=2000,
                                                      transform=transforms.Compose([transforms.ToPILImage(),
                                                                                    transforms.Resize((224, 224)),
                                                                                    transforms.ToTensor()
                                                                                    ]),
                                                      canonical=args.canonical,
                                                      noise=addnoise)
        dataset_val = teacher_tripletloss_generated(elements=200,
                                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.ToTensor(),
                                                                                  ]),
                                                    canonical=args.canonical,
                                                    noise=addnoise)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers)

        train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val)

    # List all test folders
    if args.test:
        folders = np.array([os.path.join(args.dataset, folder) for folder in os.listdir(args.dataset) if
                            os.path.isdir(os.path.join(args.dataset, folder))])

        dataset_test = teacher_tripletloss(folders, args.distance,
                                           transform=transforms.Compose([transforms.ToPILImage(),
                                                                         transforms.Resize(
                                                                             (224, 224)),
                                                                         transforms.ToTensor()
                                                                         ]),
                                           noise=addnoise,
                                           canonical=args.canonical)

        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

        # load Saved Model
        loadpath = '/home/malvaro/Documentos/IntersectionClassifier/trainedmodels/teacher/teacher_model_{}.pth'.format(
            args.resnetmodel)
        print('load model from {} ...'.format(loadpath))
        model.load_state_dict(torch.load(loadpath))
        print('Done!')

        test(args, model, dataloader_test)


def test(args, model, dataloader):
    print('\nstart test!')

    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    tq = tqdm.tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()

    all_embedding_matrix = []

    for sample in dataloader:

        if args.triplet:
            anchor = sample['anchor']  # OSM Type X
            positive = sample['positive']  # OSM Type X
            label = sample['label_anchor']
            canonical = sample['canonical']  # OSM Type X but without sampling noise (ie. angles of branches w/o noise)
        else:
            data = sample['anchor']
            label = sample['label_anchor']

        if torch.cuda.is_available() and args.use_gpu:
            if args.triplet:
                anchor = anchor.cuda()
                if args.canonical:
                    positive = canonical.cuda()
                else:
                    positive = positive.cuda()
            else:
                data = data.cuda()
                label = label.cuda()

        if args.triplet:
            out_anchor = model(anchor)
            out_positive = model(positive)
        else:
            output = model(data)

        if args.triplet:
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = ((cos(out_anchor, out_positive) + 1.0) - 0.0) * (1.0 / (2.0 - 0.0))
            probability = torch.sum(result).item()

            all_embedding_matrix.append(np.asarray(model(anchor)[0].squeeze().cpu().detach().numpy()))

            if probability >= args.threshold:
                # The prediction
                predict = label
            else:
                # The prediction is wrong, but we don't know by now what label was predicted
                predict = 7

                # Code to save the PNGs for debugging purposes
                if args.saveTestCouplesForDebug:
                    emptyspace = 255 * torch.ones([224, 30, 3], dtype=torch.float32)
                    a = plt.figure()
                    plt.imshow(np.clip(torch.cat((sample['anchor'][0].transpose(0, 2).transpose(0, 1), emptyspace,
                                                  sample['positive'][0].transpose(0, 2).transpose(0, 1), emptyspace,
                                                  torch.nn.functional.interpolate(
                                                      (sample['ground_truth_image'] / 255.0).float().transpose(1, 3),
                                                      (224, 224)).squeeze().transpose(0, 2)), 1).squeeze(), 0, 1))
                    filename = os.path.join(args.saveTestCouplesForDebugPath,
                                            str(sample['filename_anchor'][0]).split(sep="/")[6] + "-" +
                                            str(sample['filename_anchor'][0]).split(sep="/")[8])
                    plt.savefig(filename)
                    print(filename)
                    plt.close('all')

            labelRecord = np.append(labelRecord, label)
            predRecord = np.append(predRecord, predict)

        else:  # ie, not triplet
            predict = torch.argmax(output, 1)
            label = label.cpu().numpy()
            predict = predict.cpu().numpy()

            labelRecord = np.append(labelRecord, label)
            predRecord = np.append(predRecord, predict)

        tq.update(1)

    tq.close()

    # Calculate test metrics
    acc = accuracy_score(labelRecord, predRecord)
    print('Accuracy for test: %f\n' % acc)
    conf_matrix = pd.crosstab(labelRecord, predRecord, rownames=['Actual'], colnames=['Predicted'])
    conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6, 7], columns=[0, 1, 2, 3, 4, 5, 6, 7],
                                      fill_value=0)
    plt.figure(figsize=(10, 7))
    sn.heatmap(conf_matrix, annot=True)

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.log({"Test/Acc": acc, "conf-matrix": wandb.Image(plt)})

    if args.saveEmbeddings:
        all_embedding_matrix = np.asarray(all_embedding_matrix)
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_embedding_matrix.txt"), np.asarray(all_embedding_matrix),
                   delimiter='\t')
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_label_embedding_matrix.txt"), predRecord, delimiter='\t')


def validation(args, model, criterion, dataloader_val):
    print('\nstart val!')

    loss_record = 0.0
    acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    with torch.no_grad():
        model.eval()

        for sample in dataloader_val:
            if args.triplet:
                anchor = sample['anchor']  # OSM Type X
                positive = sample['positive']  # OSM Type X
                negative = sample['negative']  # OSM Type Y
            else:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                if args.triplet:
                    anchor = anchor.cuda()
                    positive = positive.cuda()
                    negative = negative.cuda()
                else:
                    data = data.cuda()
                    label = label.cuda()

            if args.triplet:
                out_anchor = model(anchor)
                out_positive = model(positive)
                out_negative = model(negative)
            else:
                output = model(data)

            if args.triplet:
                loss = criterion(out_anchor, out_positive, out_negative)
            else:
                loss = criterion(output, label)

            loss_record += loss.item()

            if args.triplet:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                result = ((cos(out_anchor, out_positive) + 1.0) - 0.0) * (1.0 / (2.0 - 0.0))
                acc_record += torch.sum(result).item()

            else:
                predict = torch.argmax(output, 1)
                label = label.cpu().numpy()
                predict = predict.cpu().numpy()

                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)

                acc_record += accuracy_score(label, predict)

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for test/validation : %f' % loss_val_mean)
    acc = acc_record / len(dataloader_val)
    print('Accuracy for test/validation : %f\n' % acc)

    return acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    # starting values
    acc_pre = 0.0
    loss_pre = np.inf

    # Build criterion
    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.zero_grad()
    model.train()

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.watch(model, log="all")

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        acc_record = 0.0

        for sample in dataloader_train:
            if args.triplet:
                anchor = sample['anchor']
                positive = sample['positive']
                negative = sample['negative']
            else:
                data = sample['anchor']
                label = sample['label_anchor']

            if torch.cuda.is_available() and args.use_gpu:
                if args.triplet:
                    anchor = anchor.cuda()
                    positive = positive.cuda()
                    negative = negative.cuda()
                else:
                    data = data.cuda()
                    label = label.cuda()

            if args.triplet:
                out_anchor = model(anchor)
                out_positive = model(positive)
                out_negative = model(negative)
            else:
                output = model(data)

            if args.triplet:
                loss = criterion(out_anchor, out_positive, out_negative)
            else:
                loss = criterion(output, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()

            if args.triplet:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                result = (((cos(out_anchor.squeeze(), out_positive.squeeze()) + 1.0) - 0.0) * (1.0 / (2.0 - 0.0)))
                acc_record += torch.sum(result).item()
            else:
                predict = torch.argmax(output, 1)
                label = label.cpu().numpy()
                predict = predict.cpu().numpy()

                acc_record += accuracy_score(label, predict)

        tq.close()

        # Calculate metrics
        loss_train_mean = loss_record / len(dataloader_train)
        if args.triplet:
            acc_train = acc_record / (len(dataloader_train) * args.batch_size)
        else:
            acc_train = acc_record / len(dataloader_train)
        print('loss for train : %f' % loss_train_mean)
        print('acc for train : %f' % acc_train)

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Train/loss": loss_train_mean,
                       "Train/acc": acc_train,
                       "Train/lr": lr}, step=epoch)

        if epoch % args.validation_step == 0:

            acc_val, loss_val = validation(args, model, criterion, dataloader_val)

            if (acc_pre < acc_val) or (loss_pre > loss_val):
                patience = 0
                if acc_pre < acc_val:
                    acc_pre = acc_val
                else:
                    loss_pre = loss_val

                bestModel = model.state_dict()
                print('Best global accuracy: {}'.format(acc_pre))

                if not args.nowandb:  # if nowandb flag was set, skip
                    wandb.log({"Val/loss": loss_val, "Val/Acc": acc_val}, step=epoch)

                print('Saving model: ',
                      os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(args.resnetmodel)))
                torch.save(bestModel,
                           os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(args.resnetmodel)))

            elif epoch < args.patience_start:
                patience = 0

            else:
                patience += 1
                print('Patience: {}\n'.format(patience))

        if patience >= args.patience > 0:
            break

        # optionally chance this values during training. Consider to add these values to wandb
        # dataset_train.set_rnd_angle(VALUE)
        # dataset_train.set_rnd_spatial()
        # dataset_train.set_rnd_width()
        # dataset_train.noise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Script modalities
    parser.add_argument('--train', action='store_true', help='Train/Validate the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--telegram', type=bool, default=True, help='Send info through Telegram')

    parser.add_argument('--triplet', action='store_true', help='Triplet Loss')
    parser.add_argument('--no_noise', action='store_true', help='In case you want to disable the nois injection in '
                                                                'the OSM images')

    # Script configuration / paths
    parser.add_argument('--dataset', type=str, help='path to the dataset you are using.')
    parser.add_argument('--save_model_path', type=str, default='./trainedmodels/teacher/', help='path to save model')

    parser.add_argument('--saveTestCouplesForDebug', action='store_true',
                        help='use this flag to ENABLE some log/debug :) see code!')
    parser.add_argument('--saveTestCouplesForDebugPath', type=str,
                        help='Where to save the saveTestCouplesForDebug PNGs. Required when --saveTestCouplesForDebug '
                             'is set')

    parser.add_argument('--saveEmbeddings', action='store_true', help='Save all the embeddings for debug')
    parser.add_argument('--saveEmbeddingsPath', type=str,
                        help='Where to save the Embeddings. Required when --saveEmbeddings is set')

    # Network behaviors
    parser.add_argument('--canonical', action='store_true', help='Used in TESTING to enable the creation of '
                                                                 'canonical images; this is used to level the test'
                                                                 'accuracy through different iterations, ie, '
                                                                 'having the same comparison images every run')

    # Newwork parameters
    parser.add_argument('--resnetmodel', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet50 or resnet101.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each batch')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=7, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU is used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--patience', type=int, default=-1, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=5,
                        help='Starting epoch for patience of validation. Default, 50. ')
    parser.add_argument('--margin', type=float, default=0.5, help='margin in triplet')
    parser.add_argument('--threshold', type=float, default=0.95, help='threshold to decide if the detection is correct')
    parser.add_argument('--distance', type=int, default=20, help='Distance to crossroads')

    args = parser.parse_args()

    if args.train and args.canonical:
        print("Mmmmm... please consider to change your mind! Creating the canonical images can speed-down the "
              "process. use --train without --canonical")
        exit(-1)

    if args.saveEmbeddings and not args.saveEmbeddingsPath:
        print("Parameter --saveEmbeddingsPath is REQUIRED when --saveEmbeddings is set")
        exit(-1)

    if args.saveTestCouplesForDebug and not args.saveTestCouplesForDebugPath:
        print("Parameter --saveTestCouplesForDebugPath is REQUIRED when --saveTestCouplesForDebug is set")
        exit(-1)

    main(args)

# Used paths:
# --saveEmbeddingsPath /media/augusto/500GBDISK/nn.based.intersection.classficator.data/debug
# --saveTestCouplesForDebugPath /media/augusto/500GBDISK/nn.based.intersection.classficator.data/debug/

# --saveEmbeddings
# --saveEmbeddingsPath
# /media/augusto/500GBDISK/nn.based.intersection.classficator.data/debug
# --saveTestCouplesForDebug
# --saveTestCouplesForDebugPath
# /media/augusto/500GBDISK/nn.based.intersection.classficator.data/debug/
