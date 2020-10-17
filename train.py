import argparse
import multiprocessing
import os
import pickle
import socket  # to get the machine name
import time
import warnings
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.transforms as transforms
import tqdm
import wandb
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataloaders.sequencedataloader import BaseLine, TestDataset, fromAANETandDualBisenet, fromGeneratedDataset, \
    triplet_BOO, triplet_OBB
from dataloaders.transforms import GenerateBev, GrayScale, Mirror, Normalize, Rescale, ToTensor
from dropout_models import get_resnet, get_resnext
from miscellaneous.utils import init_function, reset_wandb_env, send_telegram_message, send_telegram_picture, \
    student_network_pass, svm_data, svm_train
from model.resnet_models import Personalized, Personalized_small, get_model_resnet, get_model_resnext


def test(args, dataloader_test, classifier=None):
    print('start Test!')

    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin)
    elif args.embedding:
        criterion = torch.nn.CosineEmbeddingLoss(margin=args.margin)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.embedding and not args.svm:
        gt_list = []
        embeddings = np.loadtxt("./trainedmodels/teacher/embeddings/all_embedding_matrix.txt", delimiter='\t')
        splits = np.array_split(embeddings, 7)
        for i in range(7):
            gt_list.append(np.mean(splits[i], axis=0))
        gt_list = torch.FloatTensor(gt_list)
    else:
        gt_list = None  # This is made to better structure of the code ahead

    # Build model
    if args.resnetmodel[0:6] == 'resnet':
        model = get_model_resnet(args.resnetmodel, args.num_classes, transfer=args.transfer, pretrained=args.pretrained,
                                 embedding=(args.embedding or args.triplet or args.svm))
    elif args.resnetmodel[0:7] == 'resnext':
        model = get_model_resnext(args.resnetmodel, args.num_classes, args.transfer, args.pretrained)
    elif args.resnetmodel == 'personalized':
        model = Personalized(args.num_classes)
    else:
        model = Personalized_small(args.num_classes)

    # load Saved Model
    loadpath = args.student_path
    print('load model from {} ...'.format(loadpath))
    model.load_state_dict(torch.load(loadpath))
    print('Done!')

    # if args.embedding and not args.svm:
    # gt_model = copy.deepcopy(model)
    # gt_model.load_state_dict(torch.load(args.teacher_path))
    # gt_model.eval()

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()
        # if args.embedding and not args.svm:
        # gt_model = gt_model.cuda()

    # Start testing
    confusion_matrix, acc, _ = validation(args, model, criterion, dataloader_test,
                                          classifier=classifier, gt_list=gt_list)
    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.log({"Test/Acc": acc, "conf-matrix_test": wandb.Image(plt)})


def validation(args, model, criterion, dataloader_val, classifier=None, gt_list=None):
    print('\n>>>> start val!')

    loss_record = 0.0
    acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    with torch.no_grad():
        model.eval()

        tq = tqdm.tqdm(total=len(dataloader_val) * args.batch_size)
        tq.set_description('Validation... ')

        for sample in dataloader_val:
            acc, loss, label, predict = student_network_pass(args, sample, criterion, model,
                                                             svm=classifier, gt_list=gt_list)
            labelRecord = np.append(labelRecord, label)
            predRecord = np.append(predRecord, predict)

            loss_record += loss.item()
            acc_record += acc

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

        tq.close()

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for test/validation : %f' % loss_val_mean)

    if args.triplet:
        acc = acc_record / (len(dataloader_val) * args.batch_size)
    else:
        acc = acc_record / len(dataloader_val)
    print('Accuracy for test/validation : %f\n' % acc)

    if not args.triplet:
        conf_matrix = pd.crosstab(labelRecord, predRecord, rownames=['Actual'], colnames=['Predicted'])
        conf_matrix = conf_matrix.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=[0, 1, 2, 3, 4, 5, 6],
                                          fill_value=0)
        return conf_matrix, acc, loss_val_mean
    else:
        return None, acc, loss_val_mean


def train(args, model, optimizer, scheduler, dataloader_train, dataloader_val, acc_pre, valfolder, GLOBAL_EPOCH,
          kfold_index=None):
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
        kfold_index: this index is used to store indexed wandb per-batch metrics

    Returns:

    """
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    kfold_acc = 0.0
    kfold_loss = np.inf

    # Build loss criterion
    if args.weighted and not args.embedding:
        weights = [0.91, 0.95, 0.96, 0.84, 0.85, 0.82, 0.67]
        class_weights = torch.FloatTensor(weights).cuda()
        traincriterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        valcriterion = torch.nn.CrossEntropyLoss()
    elif args.embedding:
        if args.weighted:
            if args.lossfunction == 'SmoothL1':
                traincriterion = torch.nn.SmoothL1Loss(reduction='none')
                valcriterion = torch.nn.SmoothL1Loss(reduction='none')
            elif args.lossfunction == 'L1':
                traincriterion = torch.nn.L1Loss(reduction='none')
                valcriterion = torch.nn.L1Loss(reduction='none')
            elif args.lossfunction == 'MSE':
                traincriterion = torch.nn.MSELoss(reduction='none')
                valcriterion = torch.nn.MSELoss(reduction='none')
        else:
            if args.lossfunction == 'SmoothL1':
                traincriterion = torch.nn.SmoothL1Loss(reduction='mean')
                valcriterion = torch.nn.SmoothL1Loss(reduction='mean')
            elif args.lossfunction == 'L1':
                traincriterion = torch.nn.L1Loss(reduction='mean')
                valcriterion = torch.nn.L1Loss(reduction='mean')
            elif args.lossfunction == 'MSE':
                traincriterion = torch.nn.MSELoss(reduction='mean')
                valcriterion = torch.nn.MSELoss(reduction='mean')
    elif args.triplet:
        traincriterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2.0, reduction='mean')
        valcriterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2.0, reduction='mean')
    else:
        traincriterion = torch.nn.CrossEntropyLoss()
        valcriterion = torch.nn.CrossEntropyLoss()

    # Build gt images for validation
    if args.embedding:
        gt_list = []
        embeddings = np.loadtxt("./trainedmodels/teacher/embeddings/all_embedding_matrix.txt", delimiter='\t')
        splits = np.array_split(embeddings, 7)
        for i in range(7):
            gt_list.append((np.mean(splits[i], axis=0)))
        gt_list = torch.FloatTensor(gt_list)
    else:
        gt_list = None  # This is made to better structure of the code ahead

    # this can be used to verify the resume point
    if args.resume and False:
        confusion_matrix, acc_val, loss_val = validation(args, model, valcriterion, dataloader_val, gt_list=gt_list)
        if args.telegram:
            plt.figure(figsize=(10, 7))
            title = str(socket.gethostname()) + '\nResume: ' + str(args.start_epoch) + '\n' + str(valfolder)
            plt.title(title)
            sn.heatmap(confusion_matrix, annot=True, fmt='d')
            send_telegram_picture(plt, "Resume Plot")

    ###### model.zero_grad()
    model.train()

    if not args.nowandb:  # if nowandb flag was set, skip
        wandb.watch(model, log="all")

    current_batch = 0
    patience = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        print("\n\n===========================================================")
        print("date and time:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        with GLOBAL_EPOCH.get_lock():
            GLOBAL_EPOCH.value = epoch
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %.e' % (epoch, lr))
        loss_record = 0.0
        acc_record = 0.0

        for sample in dataloader_train:
            if args.embedding or args.triplet:
                acc, loss, _, _ = student_network_pass(args, sample, traincriterion, model, gt_list=gt_list)
            else:
                acc, loss, _, _ = student_network_pass(args, sample, traincriterion, model)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()
            acc_record += acc

            if not args.nowandb:  # if nowandb flag was set, skip
                if kfold_index is None:
                    kfold_index = 0
                    print("Warning, k-fold index is not passed. Ignore if not k-folding")

                wandb.log({"batch_training_accuracy": acc / args.batch_size,
                           "batch_training_loss": loss / args.batch_size,
                           "batch_current_batch": current_batch,
                           "batch_current_epoch": epoch,
                           "batch_kfold_index": kfold_index})

                current_batch += 1

        tq.close()

        # ReduceLROnPlateau is set after the validation block
        if args.scheduler and args.scheduler_type == 'MultiStepLR':
            print("MultiStepLR step call")
            scheduler.step()

        # Calculate metrics
        loss_train_mean = loss_record / len(dataloader_train)
        print('\nloss for train : %f' % loss_train_mean)

        if args.triplet or args.embedding:
            acc_train = acc_record / len(dataloader_train)
        else:
            acc_train = acc_record / len(dataloader_train)
        print('acc for train : %f' % acc_train)

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Train/loss": loss_train_mean,
                       "Train/acc": acc_train,
                       "Train/lr": optimizer.param_groups[0]['lr'] ,
                       "Completed epoch": epoch})

        if epoch % args.validation_step == 0:
            confusion_matrix, acc_val, loss_val = validation(args, model, valcriterion, dataloader_val, gt_list=gt_list)

            if args.scheduler_type == 'ReduceLROnPlateau':
                print("ReduceLROnPlateau step call")
                scheduler.step(loss_val)

            plt.figure(figsize=(10, 7))
            title = str(socket.gethostname()) + '\nEpoch: ' + str(epoch) + '\n' + str(valfolder)
            plt.title(title)
            sn.heatmap(confusion_matrix, annot=True, fmt='d')

            if args.telegram:
                send_telegram_picture(plt,
                                      "Epoch: " + str(epoch) +
                                      "\nLR: " + str(optimizer.param_groups[0]['lr']) +
                                      "\nacc_val: " + str(acc_val) +
                                      "\nloss_val: " + str(loss_val))

            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.log({"Val/loss": loss_val,
                           "Val/Acc": acc_val,
                           "Train/lr": optimizer.param_groups[0]['lr'] ,
                           "Completed epoch": epoch,
                           "conf-matrix_{}_{}".format(valfolder, epoch): wandb.Image(plt)})

            if (kfold_acc < acc_val) or (kfold_loss > loss_val):

                patience = 0

                if kfold_acc < acc_val:
                    kfold_acc = acc_val
                if kfold_loss > loss_val:
                    kfold_loss = loss_val

                if acc_pre < kfold_acc:
                    acc_pre = kfold_acc
                    print('Best global accuracy: {}'.format(kfold_acc))
                    if args.nowandb:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                            args.resnetmodel, epoch)))
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
                            'loss': loss,
                        }, os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                         args.resnetmodel, epoch)))
                    else:
                        print('Saving model: ',
                              os.path.join(args.save_model_path, '{}model_{}_{}.pth'.format(args.save_prefix,
                                                                                            wandb.run.id, epoch)))
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

    return acc_pre


def main(args, model=None):
    # Try to avoid randomness -- https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.sweep:
        hyperparameter_defaults = dict(batch_size=64, cuda='0', dataloader='BaseLine', decimate=1.0,
                                       distance=20.0, embedding=False, embedding_class=False, grayscale=False,
                                       lr=0.0001,
                                       margin=0.5, momentum=0.9, nowandb=False, num_classes=7, num_epochs=50,
                                       num_workers=4,
                                       optimizer='sgd', patience=-1, patience_start=50, pretrained=False,
                                       resnetmodel='resnet18', save_model_path='./trainedmodels/', scheduler=False,
                                       seed=0,
                                       sweep=True, telegram=False, train=True, test=False,
                                       transfer=False, triplet=False, use_gpu=True, validation_step=5, weighted=False,
                                       num_elements_OBB=2000)

        sweep = wandb.init(project="test-kfold", entity="chiringuito", config=hyperparameter_defaults, job_type="sweep",
                           reinit=True)

        # the part passed from wandb through the sweep does not contain the command line parameters in the "config" but
        # they are inside the "args" because the train.py was called... little mess.. update the config with the FEW values
        # you have in "args".
        # Example:
        # 2020-10-01 18:53:18,896 - wandb.wandb_agent - INFO - About to run command:
        # /usr/bin/env python train.py
        # --lr=0.00075 --optimizer=Adamax                                                      <<<< SWEEP PARAMETERS   !!!!
        # --embedding --teacher_path ./trainedmodels/teacher/teacher_model_sunny-sweep-1.pth   <<<< THESE ARE COMMANDS !!!!
        # --dataset ../DualBiSeNet/data_raw --sweep --train True                               <<<< THESE ARE COMMANDS !!!!
        # commands are not set in "sweep.config" so if you overwrite args with sweep.config, you'll loose some part of
        # the commands you want to give to the script

        sweep.config.update(args, allow_val_change=True)
        args = sweep.config  # get the config from the sweep

        sweep_id = sweep.sweep_id or "unknown"
        sweep_url = sweep._get_sweep_url()
        project_url = sweep._get_project_url()
        sweep.notes = "{}/groups/{}".format(project_url, sweep_id)
        sweep_run_name = sweep.name or sweep.id or "unknown"
        sweep.join()

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Starting from outer --for loop--. This is the config:")
        print("sweep.config:\n", sweep.config)
        print("sweep_id: ", sweep_id)  # this is the 'group'
        print("sweep_run_name: ", sweep_run_name)
        print("sweep project_name: ", sweep.project_name())
        print("sweep entity: ", sweep.entity)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # Getting the hostname to add to wandb (seem useful for sweeps)
        hostname = str(socket.gethostname())

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.seed = checkpoint['epoch']

    GLOBAL_EPOCH = multiprocessing.Value('i', args.seed)
    seed = multiprocessing.Value('i', args.seed)

    init_fn = partial(init_function, seed=seed, epoch=GLOBAL_EPOCH)

    # Accuracy accumulator
    acc = 0.0

    # create dataset and dataloader
    data_path = args.dataset

    # All sequence folders
    folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                        os.path.isdir(os.path.join(data_path, folder))])

    # Exclude test samples
    folders = folders[folders != os.path.join(data_path, '2011_09_30_drive_0028_sync')]
    test_path = os.path.join(data_path, '2011_09_30_drive_0028_sync')

    # Exclude validation samples
    folders = folders[folders != os.path.join(data_path, '2011_10_03_drive_0034_sync')]
    val_path = os.path.join(data_path, '2011_09_30_drive_0028_sync')

    if args.grayscale:
        aanetTransforms = transforms.Compose(
            [GenerateBev(decimate=args.decimate), Mirror(), Rescale((224, 224)), Normalize(), GrayScale(), ToTensor()])
        generateTransforms = transforms.Compose([Rescale((224, 224)), Normalize(), GrayScale(), ToTensor()])
    else:
        aanetTransforms = transforms.Compose(
            [GenerateBev(decimate=args.decimate), Mirror(), Rescale((224, 224)), Normalize(), ToTensor()])
        generateTransforms = transforms.Compose([Rescale((224, 224)), Normalize(), ToTensor()])
        obsTransforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            ])

    k_fold_acc_list = []

    if args.train:
        # loo = LeaveOneOut()
        # sweep_config = sweep.config

        # for train_index, val_index in loo.split(folders):

        # todo delete when ok -----| print("\n\n NOW K-FOLDING .... ", train_index, val_index)

        if args.sweep:
            print("******* BEGIN *******")
            reset_wandb_env()
            wandb_local_name = str(sweep_run_name) + "-split-" + str(val_index[0])
            print("Initializing wandb_current_run with name: ", wandb_local_name)
            wandb_id = wandb.util.generate_id()

            wandb_current_run = wandb.init(id=wandb_id, group=sweep_id, name=wandb_local_name, job_type=sweep.name,
                                           tags=["Teacher", "sweep", "class", hostname])

            # todo delete when ok -----| if "sweep" in args and args.sweep:
            # todo delete when ok -----|     print("YES IT IS A SWEEP! and should be called ---> " + wandb_local_name)
            # todo delete when ok -----|     print("YES IT IS A SWEEP! and its name is this ---> " + wandb_current_run.name)
            # todo delete when ok -----|     print("ITS RUN ID IS                           ---> " + wandb_current_run.id)
            # todo delete when ok -----|     print("ITS SWEEP ID IS                         ---> " + sweep_id)
            # todo delete when ok -----|     val_accuracy = random.random()
            # todo delete when ok -----|     wandb_current_run.log(dict(val_accuracy=val_accuracy))
            # todo delete when ok -----|     wandb_current_run.join()
            # todo delete when ok -----|     wandb_current_run.finish()
            # todo delete when ok -----|     print("END_RUN!!! MOVING TO THE NEXT ONE IN k-fold!" + wandb_local_name)
            # todo delete when ok -----| else:
            # todo delete when ok -----|     print("VERY SAD TIMES....")
            # todo delete when ok -----|
            # todo delete when ok -----| print("*******  END  *******")
            # todo delete when ok -----| continue
        else:
            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.init(project="nn-based-intersection-classficator", group=group_id, entity='chiringuito',
                           job_type="training", reinit=True)
                wandb.config.update(args)

        # train_path, val_path = folders[train_index], folders[val_index]
        train_path, val_path = np.array(folders), np.array([val_path])  # No kfold

        if args.dataloader == "fromAANETandDualBisenet":
            val_dataset = fromAANETandDualBisenet(val_path, args.distance, transform=aanetTransforms)
            train_dataset = fromAANETandDualBisenet(train_path, args.distance, transform=aanetTransforms)

        elif args.dataloader == "generatedDataset":
            val_dataset = fromGeneratedDataset(val_path, args.distance, transform=generateTransforms,
                                               loadlist=False,
                                               decimateStep=args.decimate,
                                               addGeneratedOSM=False)  # todo fix loadlist for k-fold
            train_dataset = fromGeneratedDataset(train_path, args.distance, transform=generateTransforms,
                                                 loadlist=False,
                                                 decimateStep=args.decimate,
                                                 addGeneratedOSM=False)  # todo fix loadlist for k-fold

        elif args.dataloader == "triplet_OBB":

            print("\nCreating train dataset from triplet_OBB")
            train_dataset = triplet_OBB(train_path, args.distance, elements=args.num_elements_OBB, canonical=False,
                                        transform_obs=obsTransforms, transform_bev=generateTransforms,
                                        loadlist=False, decimateStep=args.decimate)

            print("\nCreating validation dataset from triplet_OBB")
            val_dataset = triplet_OBB(val_path, args.distance, elements=args.num_elements_OBB, canonical=False,
                                      transform_obs=obsTransforms, transform_bev=generateTransforms, loadlist=False,
                                      decimateStep=args.decimate)

        elif args.dataloader == "triplet_BOO":

            val_dataset = triplet_BOO(val_path, args.distance, canonical=False,
                                      transform_obs=obsTransforms, transform_bev=generateTransforms,
                                      decimateStep=args.decimate)

            train_dataset = triplet_BOO(train_path, args.distance, canonical=False,
                                        transform_obs=obsTransforms, transform_bev=generateTransforms,
                                        decimateStep=args.decimate)

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
                                      num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=True)

        # Build model
        if args.resnetmodel[0:6] == 'resnet':
            model = get_model_resnet(args.resnetmodel, args.num_classes, transfer=args.transfer,
                                     pretrained=args.pretrained,
                                     embedding=(args.embedding or args.triplet or args.freeze) and not args.embedding_class)
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

        if args.freeze:
            # load best trained model
            if args.nowandb:
                loadpath = './trainedmodels/model_' + args.resnetmodel + '.pth'
            else:
                loadpath = './trainedmodels/model_' + wandb.run.id + '.pth'
            model.load_state_dict(torch.load(loadpath))
            for param in model.parameters():
                param.requires_grad = False
            model = torch.nn.Sequential(model, torch.nn.Linear(512, 7))

        if args.resume:
            if os.path.isfile(args.resume):
                # device = torch.device('cpu')
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
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
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
        else:
            print('not supported optimizer \n')
            exit()

        # Build scheduler
        if args.scheduler:
            if args.scheduler_type == 'MultiStepLR':
                if args.resume:
                    param_epoch = checkpoint['epoch']
                else:
                    param_epoch = -1
                scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 18, 20], gamma=0.5, last_epoch=param_epoch)
            if args.scheduler_type == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001,
                                              threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08, verbose=True)
            # Load Scheduler if exist
            if args.resume and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        else:
            scheduler = None

        # train model
        # acc = train(args, model, optimizer, dataloader_train, dataloader_val, acc,
        # os.path.basename(val_path[0]), GLOBAL_EPOCH=GLOBAL_EPOCH, kfold_index=val_index[0])

        ##########################################
        #   _____  ____      _     ___  _   _    #
        #  |_   _||  _ \    / \   |_ _|| \ | |   #
        #    | |  | |_) |  / _ \   | | |  \| |   #
        #    | |  |  _ <  / ___ \  | | | |\  |   #
        #    |_|  |_| \_\/_/   \_\|___||_| \_|   #
        ##########################################

        acc = train(args, model, optimizer, scheduler, dataloader_train, dataloader_val, acc,
                    os.path.basename(val_path[0]), GLOBAL_EPOCH=GLOBAL_EPOCH)

        k_fold_acc_list.append(acc)

        if args.telegram:
            send_telegram_message("K-Fold finished")

        if args.sweep:
            wandb_current_run.join()
        else:
            if not args.nowandb:  # if nowandb flag was set, skip
                wandb.join()

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.log({"Val/mean acc": np.average(np.array(k_fold_acc_list))})

        # todo delete when ok -----| print("==============================================================================")
        # todo delete when ok -----| print("=============the end of the test ===eh===eh===eh=====:-)======================")
        # todo delete when ok -----| print("==============================================================================")
        # todo delete when ok -----| sweep.join()
        # todo delete when ok -----| exit(-2)

        if args.svm:
            # load best trained model
            if args.nowandb:
                loadpath = './trainedmodels/model_' + args.resnetmodel + '.pth'
            else:
                loadpath = './trainedmodels/model_' + wandb.run.id + '.pth'
            model.load_state_dict(torch.load(loadpath))
            # save the model to disk
            embeddings, labels = svm_data(args, model, dataloader_train, dataloader_val)
            model_svm = svm_train(embeddings, labels, mode='rbf')  # embeddings: (Samples x Features); labels(Samples)
            filename = os.path.join(args.save_model_path, 'svm_classsifier.sav')
            pickle.dump(model_svm, open(filename, 'wb'))

    if args.test:
        # Final Test on 2011_10_03_drive_0027_sync
        if args.dataloader == "fromAANETandDualBisenet":
            test_dataset = TestDataset(test_path, args.distance,
                                       transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                                                          (0.229, 0.224, 0.225))
                                                                     ]))
        elif args.dataloader == 'BaseLine':
            test_dataset = BaseLine([test_path], transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))

        elif args.dataloader == 'generatedDataset':
            if args.embedding:
                test_dataset = fromGeneratedDataset(np.array([test_path]), args.distance, transform=generateTransforms)
            else:
                test_path = test_path.replace('data_raw_bev', 'data_raw')
                test_dataset = TestDataset(test_path, args.distance, transform=generateTransforms)

        elif args.dataloader == "triplet_OBB":
            test_dataset = triplet_OBB([test_path], args.distance, elements=200, canonical=True,
                                       transform_obs=obsTransforms, transform_bev=generateTransforms)
        elif args.dataloader == "triplet_BOO":
            test_dataset = triplet_BOO([test_path], args.distance, elements=200, canonical=True,
                                       transform_obs=obsTransforms, transform_bev=generateTransforms)

        dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, worker_init_fn=init_fn)

        if not args.nowandb:  # if nowandb flag was set, skip
            wandb.init(project="nn-based-intersection-classficator", group=group_id, entity='chiringuito',
                       job_type="eval")
            wandb.config.update(args)

        if args.svm:
            filename = os.path.join(args.save_model_path, 'svm_classsifier.sav')
            loaded_model = pickle.load(open(filename, 'rb'))
            test(args, dataloader_test, classifier=loaded_model)
        else:
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
    parser.add_argument('--train', action='store_true', help='Train/Validate the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--wandb_group_id', type=str, help='Set group id for the wandb experiment')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument('--sweep', action='store_true', help='if set, this run is part of a wandb-sweep; use it with'
                                                             'as documented in '
                                                             'in https://docs.wandb.com/sweeps/configuration#command')

    parser.add_argument('--telegram', action='store_true', help='Send info through Telegram')
    parser.add_argument('--freeze', action='store_true', help='fc finetuning of student model')
    parser.add_argument('--svm', action='store_true', help='support vector machine for student classification')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='Number of epochs to train for')
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
    parser.add_argument('--save_prefix', type=str, default='', help='Prefix to all saved models')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--lossfunction', type=str, default='MSE', choices=['MSE', 'SmoothL1', 'L1'],
                        help='lossfunction selection')
    parser.add_argument('--patience', type=int, default=-1, help='Patience of validation. Default, none. ')
    parser.add_argument('--patience_start', type=int, default=2,
                        help='Starting epoch for patience of validation. Default, 50. ')

    parser.add_argument('--decimate', type=int, default=1, help='How much of the points will remain after '
                                                                'decimation')
    parser.add_argument('--distance', type=float, default=20.0, help='Distance from the cross')

    parser.add_argument('--weighted', action='store_true', help='Weighted losses')
    ######    parser.add_argument('--pretrained', type=bool, default=True, help='pretrained net')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use a pretrained net, or not')
    #####    parser.add_argument('--scheduler', type=bool, default=True, help='scheduling lr')
    parser.add_argument('--scheduler', action='store_true', help='scheduling lr')
    parser.add_argument('--scheduler_type', type=str, default='MultiStepLR', choices=['MultiStepLR',
                                                                                      'ReduceLROnPlateau'])
    parser.add_argument('--grayscale', action='store_true', help='Use Grayscale Images')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint model')

    # to enable the STUDENT training, set --embedding and provide the teacher path
    parser.add_argument('--embedding', action='store_true', help='Use embedding matching')
    parser.add_argument('--embedding_class', action='store_true', help='Use embedding matching with classification')
    parser.add_argument('--triplet', action='store_true', help='Use triplet learing')
    parser.add_argument('--teacher_path', type=str, help='Insert teacher path (for student training)')
    parser.add_argument('--student_path', type=str, help='Insert student path (for student testing)')
    parser.add_argument('--margin', type=float, default=1., help='margin in triplet and embedding')

    # different data loaders, use one from choices; a description is provided in the documentation of each dataloader
    parser.add_argument('--dataloader', type=str, default='generatedDataset', choices=['fromAANETandDualBisenet',
                                                                                       'generatedDataset',
                                                                                       'BaseLine',
                                                                                       'triplet_OBB',
                                                                                       'triplet_BOO',
                                                                                       'TestDataset'],
                        help='One of the supported datasets')
    parser.add_argument('--num_elements_OBB', type=int, default=2000, help='Number of OSM in OBB training')

    args = parser.parse_args()

    # check whether --embedding was set but with no teacher path
    # if args.embedding and not args.teacher_path:
    # print("Parameter --teacher_path is REQUIRED when --embedding is set")
    # exit(-1)

    if args.svm and not args.embedding:
        print("Parameter --embedding is REQUIRED when --svm is set")

    if args.dataset == "":
        print("Empty path. Please provide the path of the dataset you want to use."
              "Ex: --dataset=../DualBiSeNet/data_raw")
        exit(-1)

    if args.triplet != (args.dataloader == 'triplet_OBB' or args.dataloader == 'triplet_BOO'):
        print("Args triplet and triplet dataloaders must be called together")
        exit(-1)

    if args.student_path:
        if not os.path.exists(args.student_path):
            print("Load file does not exist: ", args.student_path, "\n\n")
            exit(-1)

    if args.resume:
        if not os.path.exists(args.resume):
            print("checkpoint file does not exist: ", args.resume, "\n\n")
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
        group_id = 'Teacher_Student_nomask'
    print(args)
    warnings.filterwarnings("ignore")

    if args.telegram:
        send_telegram_message("Starting experiment nn-based-intersection-classficator on " + str(socket.gethostname()))

    try:
        tic = time.time()
        main(args)
        toc = time.time()
        if args.telegram:
            send_telegram_message("Experiment of nn-based-intersection-classficator ended after " +
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
            send_telegram_message("Error catched in nn-based-intersection-classficator :" + str(e) + "\nRun was on: " +
                                  str(socket.gethostname()))
