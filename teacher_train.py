import argparse
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
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from dataloaders.sequencedataloader import teacher_tripletloss, teacher_tripletloss_generated
from miscellaneous.utils import init_function, send_telegram_picture, teacher_network_pass
from model.resnet_models import get_model_resnet, get_model_vgg
from scripts.OSM_generator import test_crossing_pose

warnings.filterwarnings("ignore")


def main(args):
    # Try to avoid randomness -- https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    hyperparameter_defaults = dict(batch_size=64, canonical=False, cuda='0', dataset='../DualBiSeNet/data_raw/',
                                   dataset_train_elements=2000, dataset_val_elements=200, distance=20,
                                   enable_random_rate=True, lr=0.0001, margin=1, momentum=0.9, no_noise=False,
                                   nowandb=False, num_classes=7, num_epochs=50, num_workers=4, optimizer='sgd',
                                   patience=2, patience_start=6, pretrained=True, model='resnet18',
                                   saveEmbeddings=False, saveEmbeddingsPath=None, saveTestCouplesForDebug=False,
                                   saveTestCouplesForDebugPath=None, save_model_path='./trainedmodels/teacher/', seed=0,
                                   swap=False, telegram=True, test=False, threshold=0.92, train=True,
                                   training_rnd_angle=0.4, training_rnd_spatial=9.0, training_rnd_width=2.0,
                                   triplet=True, use_gpu=True, validation_step=5, sweep=True)

    # Getting the hostname to add to wandb (seem useful for sweeps)
    hostname = str(socket.gethostname())

    GLOBAL_EPOCH = multiprocessing.Value('i', 0)
    seed = multiprocessing.Value('i', args.seed)

    init_fn = partial(init_function, seed=seed, epoch=GLOBAL_EPOCH)

    addnoise = True
    if args.no_noise:
        addnoise = False

    # if nowandb flag was set, skip
    if not args.nowandb:
        if args.sweep:
            wandb.init(project="nn-based-intersection-classficator", entity="chiringuito", group="Teacher_train_sweep",
                       job_type="sweep", tags=["Teacher", "sweep", "class", hostname],
                       config=hyperparameter_defaults)
            args = wandb.config
        else:
            if args.test:
                wandb.init(project="nn-based-intersection-classficator", entity="chiringuito",
                           group="Teacher_train_ultimate",
                           job_type="eval", tags=["Teacher", "ultimate", "class", hostname],
                           config=hyperparameter_defaults)
            else:
                wandb.init(project="nn-based-intersection-classficator", entity="chiringuito",
                           group="Teacher_train_ultimate",
                           job_type="training", tags=["Teacher", "ultimate", "class", hostname],
                           config=hyperparameter_defaults)
            wandb.config.update(args, allow_val_change=True)

    # Build Model
    if 'vgg' in args.model:
        model = get_model_vgg(args.model, args.num_classes, pretrained=args.pretrained, embedding=args.triplet)
    else:
        model = get_model_resnet(args.model, args.num_classes, pretrained=args.pretrained, greyscale=False,
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
                                                      canonical=False,
                                                      noise=addnoise)
        dataset_val = teacher_tripletloss_generated(elements=args.dataset_val_elements,
                                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.ToTensor(),
                                                                                  ]),
                                                    canonical=False,
                                                    noise=addnoise)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, worker_init_fn=init_fn)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, worker_init_fn=init_fn)

        # Create ground truth list
        gt_list = []
        obsTransforms = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        for crossing_type in range(7):
            gt_OSM = test_crossing_pose(crossing_type=crossing_type, save=False, noise=True, sampling=False,
                                        random_rate=1.0)
            gt_OSM = obsTransforms(gt_OSM[0])
            gt_list.append(gt_OSM.unsqueeze(0))

        savepath = train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val,
                         GLOBAL_EPOCH, gt_list)

    # List all test folders
    if args.test:
        # Create ground truth list
        gt_list = []
        obsTransforms = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        for crossing_type in range(7):
            gt_OSM = test_crossing_pose(crossing_type=crossing_type, save=False, noise=True, sampling=False,
                                        random_rate=1.0)
            gt_OSM = obsTransforms(gt_OSM[0])
            gt_list.append(gt_OSM.unsqueeze(0))

        if args.testdataset == 'osm':
            folders = np.array([os.path.join(args.dataset, folder) for folder in os.listdir(args.dataset) if
                                os.path.isdir(os.path.join(args.dataset, folder))])

            dataset_test = teacher_tripletloss(folders, args.distance,
                                               transform=transforms.Compose([transforms.ToPILImage(),
                                                                             transforms.Resize(
                                                                                 (224, 224)),
                                                                             transforms.ToTensor()
                                                                             ]),
                                               noise=addnoise,
                                               canonical=True)
        else:
            dataset_test = teacher_tripletloss_generated(elements=args.dataset_test_elements,
                                                         transform=transforms.Compose([transforms.ToPILImage(),
                                                                                       transforms.Resize((224, 224)),
                                                                                       transforms.ToTensor()
                                                                                       ]),
                                                         canonical=False,
                                                         noise=addnoise)

        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

        # load Saved Model
        if args.train:
            loadpath = savepath
            args.canonical = True
        else:
            loadpath = args.trainedmodelpath

        print('load model from {} ...'.format(loadpath))
        model.load_state_dict(torch.load(loadpath))
        print('Done!')

        test(args, model, dataloader_test, gt_list)


def test(args, model, dataloader, gt_list):
    tic = time.time()
    print('\nstart test!' + str(time.strftime("%H:%M:%S", time.gmtime(tic))))

    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    tq = tqdm.tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()

    all_embedding_matrix = []
    testcriterion = torch.nn.SmoothL1Loss(reduction='mean')

    for sample in dataloader:

        if args.triplet:
            anchor = sample['anchor']  # OSM Type X
            label = sample['label_anchor']
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
            else:
                data = data.cuda()
                label = label.cuda()

        if args.triplet:
            l = []
            out_anchor = model(anchor)
            for gt in gt_list:  # Compare with the 7 canoncial ground truth ONLY WORKS WITH BATCH 1
                gt = gt.cuda()
                gt_prediction = model(gt)
                l.append(testcriterion(out_anchor, gt_prediction).item())
            nplist = np.array(l)
            nplist = nplist.reshape(-1, 7)
            predict = np.argmin(nplist, axis=1)

        else:
            output = model(data)

        if args.triplet:

            all_embedding_matrix.append(np.asarray(out_anchor.squeeze().cpu().detach().numpy()))

            # Code to save the PNGs for debugging purposes
            if args.saveTestCouplesForDebug:
                emptyspace = 255 * torch.ones([224, 30, 3], dtype=torch.float32)
                plt.figure()
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
    conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6],
                                      fill_value=0)
    plt.figure(figsize=(10, 7))
    heatmap = sn.heatmap(conf_matrix, annot=True, fmt='d')  # give a name to the heatmap, so u can call telegram

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.log({"Test/acc": acc, "conf-matrix": wandb.Image(plt)})

    # This was used to show the vector in https://projector.tensorflow.org/
    if args.saveEmbeddings:
        all_embedding_matrix = np.asarray(all_embedding_matrix)
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_embedding_matrix.txt"), np.asarray(all_embedding_matrix),
                   delimiter='\t')
        np.savetxt(os.path.join(args.saveEmbeddingsPath, "all_label_embedding_matrix.txt"), labelRecord, delimiter='\t')

    toc = time.time()

    if args.telegram:
        send_telegram_picture(heatmap.get_figure(), "Test executed in " + str(time.strftime("%H:%M:%S",
                                                                                            time.gmtime(toc - tic))) +
                              "\nAccuracy: " + "{:.4f}".format(acc) + "\nTriplet: " + str(args.triplet) +
                              "\nCanonical: " + str(args.canonical) +
                              "\nThreshold: " + str(args.threshold))
        plt.close('all')


def validation(args, model, criterion, dataloader_val, random_rate, gtlist=None):
    tic = time.time()
    print('\nstart val!\n' + str(time.strftime("%H:%M:%S", time.gmtime(tic))))

    loss_record = 0.0
    acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    with torch.no_grad():
        model.eval()

        # Optionally update the random rate for teacher_tripletloss_generated
        if args.enable_random_rate:
            dataloader_val.dataset.set_random_rate(random_rate)  # variable is wandb-tracked in train routine

        tq = tqdm.tqdm(total=len(dataloader_val) * args.batch_size)
        tq.set_description('Validation... ')

        for sample in dataloader_val:
            # network pass for the sample
            if args.triplet:
                sample_acc, loss, label, predict = teacher_network_pass(args, sample, model, criterion, gt_list=gtlist)
                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)
            else:
                sample_acc, loss, label, predict = teacher_network_pass(args, sample, model, criterion)
                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)

            loss_record += loss.item()
            acc_record += sample_acc

            tq.update(args.batch_size)

    tq.close()

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('Loss for test/validation : %f' % loss_val_mean)
    if args.triplet:
        acc = acc_record / len(dataloader_val)
    else:
        acc = acc_record / len(dataloader_val)
    print('Accuracy for test/validation : %f\n' % acc)

    conf_matrix = pd.crosstab(labelRecord, predRecord, rownames=['Actual'], colnames=['Predicted'])
    conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6],
                                      fill_value=0)

    return conf_matrix, acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val, dataset_train, dataset_val, GLOBAL_EPOCH, gtlist):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    # starting values
    acc_pre = 0.0
    loss_pre = np.inf
    random_rate = 1.0  # the random rate of the teacher_tripletloss_generated

    # Build criterion
    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=1.0, reduction='mean')
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
            if args.triplet:
                sample_acc, loss = teacher_network_pass(args, sample, model, criterion)
            else:
                sample_acc, loss, _, _ = teacher_network_pass(args, sample, model, criterion)

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
            if args.triplet:
                confusion_matrix, acc_val, loss_val = validation(args, model, criterion, dataloader_val,
                                                                 random_rate=current_random_rate, gtlist=gtlist)
            else:
                confusion_matrix, acc_val, loss_val = validation(args, model, criterion, dataloader_val,
                                                                 random_rate=current_random_rate)

            if (acc_pre < acc_val) or (loss_pre > loss_val):
                patience = 0
                if acc_pre < acc_val:
                    acc_pre = acc_val
                if loss_pre > loss_val:
                    loss_pre = loss_val

                bestModel = model.state_dict()
                print('Best global accuracy: {}'.format(acc_pre))

                if not args.nowandb:  # if nowandb flag was set, skip
                    plt.figure(figsize=(10, 7))
                    sn.heatmap(confusion_matrix, annot=True, fmt='d')
                    wandb.log({"Val/loss": loss_val,
                               "Val/Acc": acc_val,
                               "random_rate": random_rate,
                               "conf-matrix_{}_{}".format(wandb.run.name, epoch): wandb.Image(plt)}, step=epoch)
                if args.nowandb:
                    if args.triplet:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(args.model)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_{}.pth'.format(args.model)))
                        savepath = os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(args.model))
                    else:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(args.model)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_class_{}.pth'.format(args.model)))
                        savepath = os.path.join(args.save_model_path,
                                                'teacher_model_class_{}.pth'.format(args.model))
                else:
                    if args.triplet:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(wandb.run.name)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_{}.pth'.format(wandb.run.name)))
                        savepath = os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(wandb.run.name))
                    else:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, 'teacher_model_class_{}.pth'.format(wandb.run.name)))
                        torch.save(bestModel, os.path.join(args.save_model_path,
                                                           'teacher_model_class_{}.pth'.format(wandb.run.name)))
                        savepath = os.path.join(args.save_model_path,
                                                'teacher_model_class_{}.pth'.format(wandb.run.name))

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

    return savepath


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
    parser.add_argument('--train', action='store_true', help='Train/Validate the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--sweep', action='store_true', help='if set, this run is part of a wandb-sweep; use it with'
                                                             'as documented in '
                                                             'in https://docs.wandb.com/sweeps/configuration#command')
    parser.add_argument('--telegram', action='store_true', help='Send info through Telegram')

    parser.add_argument('--triplet', type=bool, default=True, help='Triplet Loss')
    parser.add_argument('--swap', action='store_true', help='Triplet Loss swap')
    parser.add_argument('--margin', type=float, default=2.0, help='margin in triplet')
    parser.add_argument('--no_noise', action='store_true', help='In case you want to disable the noise injection in '
                                                                'the OSM images')

    parser.add_argument('--canonical', action='store_true', help='Used in TESTING to enable the creation of '
                                                                 'canonical images; this is used to level the test'
                                                                 'accuracy through different iterations, ie, '
                                                                 'having the same comparison images every run')

    parser.add_argument('--testdataset', type=str, default='osm', choices=['osm', 'generated'], help='dataloader for test')
    parser.add_argument('--dataset_train_elements', type=int, default=2000, help='see teacher_tripletloss_generated')
    parser.add_argument('--dataset_val_elements', type=int, default=100, help='see teacher_tripletloss_generated')
    parser.add_argument('--dataset_test_elements', type=int, default=1000, help='see teacher_tripletloss_generated')
    parser.add_argument('--training_rnd_width', type=float, default=2.0, help='see teacher_tripletloss_generated')
    parser.add_argument('--training_rnd_angle', type=float, default=0.4, help='see teacher_tripletloss_generated')
    parser.add_argument('--training_rnd_spatial', type=float, default=9.0, help='see teacher_tripletloss_generated')
    parser.add_argument('--enable_random_rate', type=bool, default=True, help='see teacher_tripletloss_generated')

    ################################
    # SCRIPT CONFIGURATION / PATHS #
    ################################
    parser.add_argument('--dataset', default='../DualBiSeNet/data_raw/', type=str,
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
    parser.add_argument('--trainedmodelpath', type=str,
                        help='Path where the weights of the trained model were saved')

    #####################################
    # NETWORK PARAMETERS (FOR BACKBONE) #
    #####################################
    parser.add_argument('--model', type=str, default="resnet18",
                        choices=['resnet18', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='The context path model you are using, resnet18, resnet50 or resnet101.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each batch')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=2, help='How often to perform validation and a '
                                                                       'checkpoint (epochs)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=7, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU is used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--patience', type=int, default=2, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=2,
                        help='Starting epoch for patience of validation. Default, 50. ')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use a pretrained net, or not')
    parser.add_argument('--threshold', type=float, default=0.92, help='threshold to decide if the detection is correct')
    parser.add_argument('--distance', type=int, default=20, help='Distance to crossroads')

    args = parser.parse_args()

    if args.train and args.canonical:
        print("Mmmmm... please consider to change your mind! Creating the canonical images can speed-down the "
              "process. use --train without --canonical")
        exit(-1)

    if args.test and not args.train and not args.trainedmodelpath:
        print("Mmmmm... please consider to select a trained network to test.")
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

    print(args)
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
