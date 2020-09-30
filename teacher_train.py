import socket  # to get the machine name
import multiprocessing
from functools import partial

import argparse
import os
import numpy as np
import tqdm
import pandas as pd

import torchvision.transforms as transforms
from dataloaders.sequencedataloader import teacher_tripletloss_generated, teacher_tripletloss

import torch
from torch.utils.data import DataLoader
from torch import nn

from model.resnet_models import get_model_resnet

from sklearn.metrics import accuracy_score

from miscellaneous.utils import send_telegram_picture, teacher_network_pass
import time

import matplotlib.pyplot as plt

import wandb
import seaborn as sn

import random

import warnings
warnings.filterwarnings("ignore")

def _init_fn(worker_id, seed, epoch):
    seed = seed.value + worker_id + epoch.value * 100
    # if you want to debug... print(f"\nInit worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):

    # Try to avoid randomness -- https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    hyperparameter_defaults = dict(batch_size=64, canonical=False, cuda='0', dataset='../DualBiSeNet/data_raw_bev/',
                                   dataset_train_elements=2000, dataset_val_elements=200, distance=20,
                                   enable_random_rate=True, lr=0.0001, margin=1, momentum=0.9, no_noise=False,
                                   nowandb=False, num_classes=7, num_epochs=50, num_workers=4, optimizer='sgd',
                                   patience=2, patience_start=6, pretrained=True, resnetmodel='resnet18',
                                   saveEmbeddings=False, saveEmbeddingsPath=None, saveTestCouplesForDebug=False,
                                   saveTestCouplesForDebugPath=None, save_model_path='./trainedmodels/teacher/', seed=0,
                                   swap=False, telegram=True, test=False, threshold=0.92, train=True,
                                   training_rnd_angle=0.4, training_rnd_spatial=9.0, training_rnd_width=2.0,
                                   triplet=True, use_gpu=True, validation_step=5)

    # Getting the hostname to add to wandb (seem useful for sweeps)
    hostname = str(socket.gethostname())

    GLOBAL_EPOCH = multiprocessing.Value('i', 0)
    seed = multiprocessing.Value('i', args.seed)

    init_fn = partial(_init_fn, seed=seed, epoch=GLOBAL_EPOCH)

    addnoise = True
    if args.no_noise:
        addnoise = False

    # if nowandb flag was set, skip
    if not args.nowandb:
        if args.test:
            wandb.init(project="nn-based-intersection-classficator", entity="chiringuito", group="Teacher_train_sweep",
                       job_type="eval", tags=["Teacher", "sweep", hostname])
        else:
            wandb.init(project="nn-based-intersection-classficator", entity="chiringuito", group="Teacher_train_sweep",
                       job_type="training", tags=["Teacher", "sweep", hostname], config=hyperparameter_defaults)
        args = wandb.config
        #wandb.config.update(args, allow_val_change=True)

    # Build Model
    model = get_model_resnet(args.resnetmodel, args.num_classes, pretrained=args.pretrained, greyscale=False,
                             embedding=args.triplet)

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    if args.train:
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

        # In both, set canonical to False to speedup the process; canonical won't be used for train/validate.
        dataset_train = teacher_tripletloss_generated(elements=args.dataset_train_elements,
                                                      transform=transforms.Compose([transforms.ToPILImage(),
                                                                                    transforms.Resize((224, 224)),
                                                                                    transforms.ToTensor()
                                                                                    ]),
                                                      canonical=args.canonical,
                                                      noise=addnoise)
        dataset_val = teacher_tripletloss_generated(elements=args.dataset_val_elements,
                                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.ToTensor(),
                                                                                  ]),
                                                    canonical=args.canonical,
                                                    noise=addnoise)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, worker_init_fn=init_fn)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, worker_init_fn=init_fn)

        train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val, GLOBAL_EPOCH)

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
        if args.nowandb:
            if args.triplet:
                loadpath = './trainedmodels/teacher/teacher_model_{}.pth'.format(args.resnetmodel)
            else:
                loadpath = './trainedmodels/teacher/teacher_model_class_{}.pth'.format(args.resnetmodel)
        else:
            if args.triplet:
                loadpath = './trainedmodels/teacher/teacher_model_{}.pth'.format(wandb.run.name)
            else:
                loadpath = './trainedmodels/teacher/teacher_model_class_{}.pth'.format(wandb.run.name)

        print('load model from {} ...'.format(loadpath))
        model.load_state_dict(torch.load(loadpath))
        print('Done!')

        test(args, model, dataloader_test)


def test(args, model, dataloader):
    tic = time.time()
    print('\nstart test!' + str(time.strftime("%H:%M:%S", time.gmtime(tic))))

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

        # if args.canonical:
        #    a = plt.figure()
        #    plt.imshow(canonical.squeeze().numpy().transpose((1, 2, 0)))
        #    send_telegram_picture(a, "canonical")
        #    plt.close('all')

        # a = plt.figure()
        # plt.imshow(sample['anchor'].squeeze().numpy().transpose((1, 2, 0)))
        # send_telegram_picture(a, "anchor")
        # plt.close('all')

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
            result = (cos(out_anchor, out_positive) + 1.0) * 0.5
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
    heatmap = sn.heatmap(conf_matrix, annot=True, fmt='d')  # give a name to the heatmap, so u can call telegram

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.log({"Test/Acc": acc, "conf-matrix": wandb.Image(plt)})

    # This was used to show the vector in https://projector.tensorflow.org/
    if args.saveEmbeddings:
        all_embedding_matrix = np.asarray(all_embedding_matrix)
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_embedding_matrix.txt"), np.asarray(all_embedding_matrix),
                   delimiter='\t')
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_label_embedding_matrix.txt"), predRecord, delimiter='\t')

    toc = time.time()

    if args.telegram:
        send_telegram_picture(heatmap.get_figure(), "Test executed in " + str(time.strftime("%H:%M:%S",
                                                                                            time.gmtime(toc - tic))) +
                              "\nAccuracy: " + "{:.4f}".format(acc) + "\nTriplet: " + str(args.triplet) +
                              "\nCanonical: " + str(args.canonical) +
                              "\nThreshold: " + str(args.threshold))
        plt.close('all')


def validation(args, model, criterion, dataloader_val, random_rate):
    tic = time.time()
    print('\nstart val!\n' + str(time.strftime("%H:%M:%S", time.gmtime(tic))))

    loss_record = 0.0
    acc_record = 0.0

    with torch.no_grad():
        model.eval()

        # Optionally update the random rate for teacher_tripletloss_generated
        if args.enable_random_rate:
            dataloader_val.dataset.set_random_rate(random_rate)  # variable is wandb-tracked in train routine

        tq = tqdm.tqdm(total=len(dataloader_val) * args.batch_size)
        tq.set_description('Validation... ')

        for sample in dataloader_val:
            # network pass for the sample
            sample_acc, loss = teacher_network_pass(args, sample, model, criterion)

            loss_record += loss.item()
            acc_record += sample_acc

            tq.update(args.batch_size)

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('Loss for test/validation : %f' % loss_val_mean)
    acc = acc_record / (len(dataloader_val) * args.batch_size)
    print('Accuracy for test/validation : %f\n' % acc)

    return acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val, GLOBAL_EPOCH):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    # starting values
    acc_pre = 0.0
    loss_pre = np.inf
    random_rate = 1.0  # the random rate of the teacher_tripletloss_generated

    # Build criterion
    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin, swap=args.swap)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.zero_grad()
    model.train()

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.watch(model, log="all")

    if args.enable_random_rate:
        current_random_rate = 0.5
    else:
        current_random_rate = 1.0

    for epoch in range(args.num_epochs):
        with GLOBAL_EPOCH.get_lock():
            GLOBAL_EPOCH.value = epoch
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        acc_record = 0.0

        # Optionally update the random rate for teacher_tripletloss_generated
        if args.enable_random_rate:
            random_rate = dataloader_train.dataset.set_random_rate(current_random_rate)

        for sample in dataloader_train:
            # network pass for the sample
            sample_acc, loss = teacher_network_pass(args, sample, model, criterion)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()
            acc_record += sample_acc

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
                       "Train/lr": lr,
                       "random_rate": random_rate,
                       "Completed epoch": epoch}, step=epoch)

        if epoch % args.validation_step == 0:

            acc_val, loss_val = validation(args, model, criterion, dataloader_val, random_rate=current_random_rate)

            if (acc_pre < acc_val) or (loss_pre > loss_val):
                patience = 0
                if acc_pre < acc_val:
                    acc_pre = acc_val
                if loss_pre > loss_val:
                    loss_pre = loss_val

                bestModel = model.state_dict()
                print('Best global accuracy: {}'.format(acc_pre))

                if not args.nowandb:  # if nowandb flag was set, skip
                    wandb.log({"Val/loss": loss_val,
                               "Val/Acc": acc_val,
                               "random_rate": random_rate}, step=epoch)
                if args.nowandb:
                    if args.triplet:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(args.resnetmodel)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_{}.pth'.format(args.resnetmodel)))
                    else:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(args.resnetmodel)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_class_{}.pth'.format(args.resnetmodel)))
                else:
                    if args.triplet:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(wandb.run.name)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_{}.pth'.format(wandb.run.name)))
                    else:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(wandb.run.name)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_class_{}.pth'.format(wandb.run.name)))

            elif epoch < args.patience_start:
                patience = 0

            else:
                patience += 1
                print('Patience: {}\n'.format(patience))

        if patience >= args.patience > 0 or acc_val == 1:
            break

        if args.enable_random_rate:
            current_random_rate = current_random_rate + 0.05
            if current_random_rate >= 1.0:
                current_random_rate = 1.0
        else:
            current_random_rate = 1.0


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

    parser = argparse.ArgumentParser()

    ###########################################
    # SCRIPT MODALITIES AND NETWORK BEHAVIORS #
    ###########################################
    parser.add_argument('--seed', type=int, default=0, help='Starting seed, for reproducibility. Default is ZERO!')
    parser.add_argument('--train', type=bool, default=True, help='Train/Validate the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--telegram', type=bool, default=True, help='Send info through Telegram')

    parser.add_argument('--triplet', type=bool, default=True, help='Triplet Loss')
    parser.add_argument('--swap', action='store_true', help='Triplet Loss swap')
    parser.add_argument('--margin', type=float, default=1, help='margin in triplet')
    parser.add_argument('--no_noise', action='store_true', help='In case you want to disable the noise injection in '
                                                                'the OSM images')

    parser.add_argument('--canonical', action='store_true', help='Used in TESTING to enable the creation of '
                                                                 'canonical images; this is used to level the test'
                                                                 'accuracy through different iterations, ie, '
                                                                 'having the same comparison images every run')

    parser.add_argument('--dataset_train_elements', type=int, default=2000, help='see teacher_tripletloss_generated')
    parser.add_argument('--dataset_val_elements', type=int, default=200, help='see teacher_tripletloss_generated')

    parser.add_argument('--training_rnd_width', type=float, default=2.0, help='see teacher_tripletloss_generated')
    parser.add_argument('--training_rnd_angle', type=float, default=0.4, help='see teacher_tripletloss_generated')
    parser.add_argument('--training_rnd_spatial', type=float, default=9.0, help='see teacher_tripletloss_generated')
    parser.add_argument('--enable_random_rate', type=bool, default=True, help='see teacher_tripletloss_generated')

    ################################
    # SCRIPT CONFIGURATION / PATHS #
    ################################
    parser.add_argument('--dataset', default='../DualBiSeNet/data_raw_bev/', type=str,
                        help='path to the dataset you are using.')
    parser.add_argument('--save_model_path', type=str, default='./trainedmodels/teacher/', help='path to save model')

    parser.add_argument('--saveTestCouplesForDebug', action='store_true',
                        help='use this flag to ENABLE some log/debug :) see code!')
    parser.add_argument('--saveTestCouplesForDebugPath', type=str,
                        help='Where to save the saveTestCouplesForDebug PNGs. Required when --saveTestCouplesForDebug '
                             'is set')

    parser.add_argument('--saveEmbeddings', action='store_true', help='Save all the embeddings for debug')
    parser.add_argument('--saveEmbeddingsPath', type=str,
                        help='Where to save the Embeddings. Required when --saveEmbeddings is set')

    #####################################
    # NETWORK PARAMETERS (FOR BACKBONE) #
    #####################################
    parser.add_argument('--resnetmodel', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet50 or resnet101.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each batch')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=6, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=7, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU is used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--patience', type=int, default=2, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=6,
                        help='Starting epoch for patience of validation. Default, 50. ')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use a pretrained net, or not')
    parser.add_argument('--threshold', type=float, default=0.92, help='threshold to decide if the detection is correct')
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

    if args.swap and not args.triplet:
        print("Parameter --swap is not necessary for classification")
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
