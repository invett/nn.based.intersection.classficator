# Dataloader for DualBisenet under prepared Kitti dataset
import glob
import json
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from numpy import load
from torch.utils.data import Dataset

from miscellaneous.utils import write_ply

from random import choice
from collections import Counter

from scripts.OSM_generator import test_crossing_pose, Crossing


class AbstractSequence:
    """
    This "abstract" class is used to include/inherance in all future "sequences" dataloaders
    """

    def __init__(self, isSequence=False):
        self.isSequence = isSequence

    def getIsSequence(self):
        return self.isSequence


class txt_dataloader(AbstractSequence, Dataset):
    def __init__(self, path_filename_list=None, transform=None, usePIL=True, isSequence=False, decimateStep=1,
                 verbose=True):
        """

                THIS IS THE DATALOADER USES the split files generated with labelling-script.py

                USED TO TRAIN THE FRAME-BASED CLASSIFICATOR!

                Args:
                    path_filename_list (string): a list of filenames with all the files that you want to use;
                                                 this dataloader uses a file; to mantain compatibility with previous
                                                 versions, we alsto support for single string.
                    with all the images, does not walk a os folder!
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                    usePIL: default True, but if not, return numpy-arrays!
                    decimateStep: use this value to decimate the dataset; set as "::STEP"

                the path_filename will be used together with the lines of the file to create the full-paths of the imgs

                as example, given:

                    path_filename = '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_all.txt'

                and:
                    head /home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_all.txt
                        164057AA/0000004164.png;0
                        164057AA/0000004165.png;0
                        ...
                        164057AA/0000004167.png;0

                then, resulting filenames will be a mix between the first part of path_filename and each line

                        /home/ballardini/DualBiSeNet/alcala-26.01.2021_selected  +  164057AA/0000004164.png;0
                        /home/ballardini/DualBiSeNet/alcala-26.01.2021_selected  +  164057AA/0000004165.png;0
                        ...
                        /home/ballardini/DualBiSeNet/alcala-26.01.2021_selected  +  164057AA/0000004167.png;0

        """

        if not isinstance(decimateStep, int) and decimateStep > 0:
            print("decimateStep must be an integer > 0. Passed: ", decimateStep)
            exit(-1)

        super().__init__(isSequence=isSequence)
        self.verbose = verbose
        self.GANflag = False  # this flag is used to control the getItem method

        trainimages = []
        trainlabels = []

        if isinstance(path_filename_list, str):
            path_filename_list = [path_filename_list]

        # cycle through list of path_filename_list.. this was introduced to allow multi-dataset loadings
        for path_filename in path_filename_list:

            if self.verbose:
                print('Loading: ' + str(path_filename) + ' ...')

            # Check the file containing all the images! This dataloader does not work walking a folder!
            if not os.path.isfile(path_filename):
                print('Class: ' + __class__.__name__, " - file doesn't exist - ", path_filename)
                exit(-1)

            with open(path_filename) as filename:
                Lines = filename.readlines()
                for line in Lines:
                    # trainimages.append(os.path.join('../DualBiSeNet/', line.strip().split(';')[0]))
                    trainimages.append(os.path.join(os.path.split(path_filename)[0], line.strip().split(';')[0]))
                    trainlabels.append(line.strip().split(';')[1])

        if self.verbose:
            print('Images loaded: ' + str(len(trainimages)) + '\n')

        self.transform = transform

        # decimate
        if decimateStep != 1:
            if self.verbose:
                print("The dataset will be decimated taking 1 out of " + str(decimateStep) + " elements")
            trainimages = trainimages[::decimateStep]
            trainlabels = trainlabels[::decimateStep]

        self.images = trainimages
        self.labels = trainlabels
        self.usePIL = usePIL

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # Select file subset

        if self.GANflag:
            imagepath = self.imgs[idx][0]
            label = int(self.imgs[idx][1])
        else:
            imagepath = self.images[idx]
            label = int(self.labels[idx])

        image = Image.open(imagepath)

        neg_label = choice([i for i in range(0, 7) if i != label])

        if not self.usePIL:
            # copy to avoid warnings from pytorch, or bad edits ...
            image = np.copy(np.asarray(image))

            sample = {'image_02': image,
                      'label': label,
                      'neg_label': neg_label,
                      'path_of_original_image': imagepath}

            if self.transform:
                transformed = self.transform(sample)
            else:
                transformed = sample

            # save the image if needed, ie, we have the path inserted with the "fake-transform".
            if "path" in transformed:

                if 'KITTI-ROAD' in imagepath:
                    dataset_path = os.path.split(imagepath)[0].replace('KITTI-ROAD', os.path.split(transformed['path'])[1])
                elif 'KITTI-360' in imagepath:
                    dataset_path = os.path.split(imagepath)[0].replace('KITTI-360', os.path.split(transformed['path'])[1])
                else:
                    dataset_path = os.path.join(transformed['path'], imagepath.split('/')[-2])

                # TODO: HANDLE THIS HELL...
                dataset_path = os.path.join(transformed['path'], imagepath.split('/')[-4], 'image_02', 'data')


                if not os.path.isdir(dataset_path):
                    os.makedirs(dataset_path)
                base_file_star = os.path.splitext(os.path.split(imagepath)[1])[0]
                current_filelist = glob.glob1(dataset_path, base_file_star + '*')
                last_number = len([x for x in current_filelist if "json" not in x])
                final_filename = base_file_star + '.' + str(last_number + 1).zfill(3) + '.png'
                bev_path_filename = os.path.join(dataset_path, final_filename)

                print("Saving image in ", bev_path_filename)

                wheretowrite = os.path.join(transformed['path'], 'output.txt')
                towrite = os.path.join(os.path.split(dataset_path)[1], final_filename) + ';' + str(
                    transformed['label']) + '\n'
                with open(wheretowrite, 'a') as file_object:
                    file_object.write(towrite)

                # path must already exist!
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                flag = cv2.imwrite(bev_path_filename, transformed['data'])
                assert flag, "can't write file"

            sample = {'data': transformed['data'],
                      'label': label,
                      'neg_label': neg_label,
                      'path_of_original_image': imagepath}

        else:
            sample = {'data': image,
                      'label': label,
                      'neg_label': neg_label,
                      'path_of_original_image': imagepath}

            if self.transform:
                sample['data'] = self.transform(sample['data'])

        return sample


class kitti360(AbstractSequence, Dataset):
    def __init__(self, path, sequence_list, transform=None, isSequence=False):
        """

                THIS IS THE DATALOADER USED TO DIRECTLY USE RGB IMAGES on Kitti 360 dataset
                ALSO WORKS WITH THE KITTI 360 WARPINGS, as the folders are organized in the same way.

                Args:
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.


        """
        super().__init__(isSequence=isSequence)
        self.transform = transform

        images = {}
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                # name example: '2013_05_28_drive_0002_sync_0000018453.png'
                head, ext = os.path.splitext(name)
                if (ext == '.png') and (root.split('/')[-1] == 'left'):
                    # sequence example: 2013_05_28_drive_0002_sync
                    sequence = '_'.join(name.split('_')[0:6])  # kitti360-augusto
                    # sequence = '_'.join(name.split('_')[0:1])  # alcala26.01.21

                    # frame example: 0000018453.png
                    frame = name.split('_')[-1]

                    # label example: 2 (from the folder name)
                    label = int(root.split('/')[-2])

                    # this if appends or creates the first element of the list
                    if sequence in images:
                        images[sequence]['labels'].append(label)
                        images[sequence]['frames'].append(os.path.join(root, '_'.join([sequence, frame])))
                    else:
                        images[sequence] = {'labels': [label],
                                            'frames': [os.path.join(root, '_'.join([sequence, frame]))]}

        trainimages = []
        trainlabels = []

        # images should contain a dictionary with
        # 2013_05_28_drive_0003_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0002_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0005_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0006_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0007_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0009_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0010_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0004_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        # 2013_05_28_drive_0000_sync = {dict: 2} {'labels': [..., ...], 'frames': ['../DualBiseNet/kitti/5/left/.png']
        for sequence, samples in images.items():
            if sequence in sequence_list:
                for k, list in samples.items():
                    if k == 'labels':
                        trainlabels.extend(list)
                    if k == 'frames':
                        trainimages.extend(list)

        self.images = trainimages
        self.labels = trainlabels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # Select file subset
        imagepath = self.images[idx]
        image = Image.open(imagepath)

        label = self.labels[idx]
        neg_label = choice([i for i in range(0, 7) if i != label])

        sample = {'data': image,
                  'label': label,
                  'neg_label': neg_label}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample


class Kitti2011_RGB(Dataset):
    def __init__(self, folders, transform=None):
        """

        THIS IS THE DATALOADER USED TO DIRECTLY USE RGB IMAGES

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform

        image_02 = []

        for folder in folders:
            folder_image_02 = os.path.join(folder, 'image_02')
            for image_02_file in os.listdir(folder_image_02):
                _, ext = os.path.splitext(image_02_file)
                if os.path.isfile(os.path.join(folder_image_02, image_02_file)) and (ext == '.png'):
                    image_02.append(os.path.join(folder_image_02, image_02_file))

        self.image_02 = image_02

        assert len(self.image_02) > 0, 'Training files missing'

    def __len__(self):

        return len(self.image_02)

    def __getitem__(self, idx):
        # Select file subset
        imagepath = self.image_02[idx]

        image = Image.open(imagepath)

        # Obtaining ground truth
        if os.path.isfile(imagepath + '.json'):
            with open(imagepath + '.json') as json_file:
                gTruth_info = json.load(json_file)
                gTruth = int(gTruth_info['label'])
        else:
            head, tail = os.path.split(imagepath)
            head, _ = os.path.split(head)
            filename, _ = os.path.splitext(tail)
            gt_path = os.path.join(head, 'frames_topology.txt')
            gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
            gTruth = int(gtdata.loc[gtdata[0] == filename][2])

        sample = {'data': image,
                  'label': gTruth}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample


class TestDataset(Dataset):

    def __init__(self, root_dir, distance=None, transform=None):
        """

        THIS IS THE FIRST DATA LOADER WE USED WITH THE BEVs GENERATED OFFLINE WITH THE AUGUSTO's SCRIPTS, NOT INSIDE
        THIS PROJECT. THE DATASET DIDN'T CONTAIN ANY DATA AUGMENTATION, WE MADE DATA AUGMENTATION HERE BUT IN THIS WAY
        THE NUMBER OF POINTS CAN NOT CHANGE FOR EXAMPLE IN A ROTATION, RESULTING IN A BLACK AREA OF THE IMAGE WHERE
        INSTEAD WE CAN DO WAY BETTER GENERATING ON-THE-FLY THE BEVs.

        THIS WAS DONE WITH <fromAANETandDualBisenet> DATALOADER, BUT IT DRASTICALLY DECREASES THE SPEED OF THE PROCESS.
        FOR THIS REASON WE MADE AN "AUGMENTED" OFFLINE DATASET WITH A NEW DATALOADER, <fromGeneratedDataset>

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform

        root_dir = os.path.join(root_dir, 'bev')

        files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                 os.path.isfile(os.path.join(root_dir, name)) and '.png' in name]
        self.file_list = files
        if distance is not None:
            self.__filterdistance(distance)
        assert len(self.file_list) > 0, 'Training files missing'

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Select file subset
        imagepath = self.file_list[idx]

        image = Image.open(imagepath)

        # Obtaining ground truth
        head, tail = os.path.split(imagepath)
        head, _ = os.path.split(head)
        filename, _ = os.path.splitext(tail)
        gt_path = os.path.join(head, 'frames_topology.txt')
        gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        gTruth = int(gtdata.loc[gtdata[0] == filename][2])

        sample = {'data': image, 'label': gTruth}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample

    def __filterdistance(self, distance):
        images = []
        for file in self.file_list:
            head, filename = os.path.split(file)
            head, _ = os.path.split(os.path.normpath(head))
            datapath = os.path.join(head, 'frames_topology.txt')
            name, _ = os.path.splitext(filename)
            gtdata = pd.read_csv(datapath, sep=';', header=None, dtype=str)
            if float(gtdata.loc[gtdata[0] == name][1]) < distance:
                images.append(file)

        self.file_list = images


class fromAANETandDualBisenet(Dataset):

    def __init__(self, folders, distance, transform=None):
        """

        RUNTIME Dataloader that uses grounth truth dataset, the subset of the original RGB KITTI images, and the results
        from AANET and DualBiseNet to create augmented-images by using a 3D point cloud generated from AANET. Please
        notice that the "real" 3D work is performed in the 'GenerateBev' transform!

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.


        'pred' --> contains the AANET npz files; the names are like 0000000000_pred.npz
        'alvaromask' --> contains the DualBisnet output files; filename format is like 0000000000pred.png

        """

        self.transform = transform

        aanet = []
        alvaromask = []
        image_02 = []

        for folder in folders:
            folder_aanet = os.path.join(folder, 'pred')
            folder_alvaromask = os.path.join(folder, 'alvaromask')
            folder_image_02 = os.path.join(folder, 'image_02')
            for file in os.listdir(folder_aanet):
                alvarofile = file.replace("_pred.npz", "pred.png")
                image_02_file = file.replace("_pred.npz", ".png")

                if os.path.isfile(os.path.join(folder_aanet, file)) and os.path.isfile(
                        os.path.join(folder_alvaromask, alvarofile)) and os.path.isfile(
                    os.path.join(folder_image_02, image_02_file)):
                    aanet.append(os.path.join(folder_aanet, file))
                    alvaromask.append(os.path.join(folder_alvaromask, alvarofile))
                    image_02.append(os.path.join(folder_image_02, image_02_file))
                else:
                    print("Loader error")
                    print(os.path.join(folder_aanet, file))
                    print(os.path.join(folder_alvaromask, alvarofile))
                    print(os.path.join(folder_image_02, image_02_file))

        self.aanet = aanet
        self.alvaromask = alvaromask
        self.image_02 = image_02

        self.__filterdistance(distance)

        assert len(self.aanet) > 0, 'Training files missing [aanet]'
        assert len(self.alvaromask) > 0, 'Training files missing [alvaromask]'
        assert len(self.image_02) > 0, 'Training files missing [alvaromask]'

    def __len__(self):

        return len(self.aanet)

    def __getitem__(self, idx):

        # Select file subset
        aanet_file = self.aanet[idx]
        alvaromask_file = self.alvaromask[idx]
        image_02_file = self.image_02[idx]

        dict_data = load(aanet_file)
        aanet_image = dict_data['arr_0']
        alvaromask_image = cv2.imread(alvaromask_file, cv2.IMREAD_UNCHANGED)
        image_02_image = cv2.imread(image_02_file, cv2.IMREAD_UNCHANGED)

        # Obtaining ground truth
        head, tail = os.path.split(aanet_file)
        head, _ = os.path.split(head)
        filename, _ = os.path.splitext(tail)
        gt_path = os.path.join(head, 'frames_topology.txt')
        gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        gTruth = int(gtdata.loc[gtdata[0] == filename.replace("_pred", "")][2])

        sample = {'aanet': aanet_image,
                  'alvaromask': alvaromask_image,
                  'image_02': image_02_image,
                  'label': gTruth}

        assert self.transform, "no transform list provided"

        # call all the transforms
        bev_with_new_label = self.transform(sample)

        # check whether the <GenerateNewDataset> transform was included (check the path); if yes, save the generated BEV
        # please notice that the path.parameter isn't actually used, was our first intention, but then was simpler to do
        # the following... OPTIMIZE change "path" as string with boolean
        if "path" in bev_with_new_label:

            folder, file = os.path.split(image_02_file.replace("data_raw", "data_raw_bev").replace("image_02", ""))
            dataset_path = os.path.join(bev_with_new_label['path'], os.path.basename(folder))
            base_file_star = str(file.split(".")[0]) + "*"
            current_filelist = glob.glob1(dataset_path, base_file_star)
            last_number = len([x for x in current_filelist if "json" not in x])
            final_filename = str(file.split(".")[0]) + '.' + str(last_number + 1).zfill(3) + ".png"
            bev_path_filename = os.path.join(dataset_path, final_filename)

            # path must already exist!
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            flag = cv2.imwrite(bev_path_filename, bev_with_new_label['data'])
            assert flag, "can't write file"
            bev_with_new_label['bev_path_filename'] = bev_path_filename

            # for debuggin' purposes
            if "save_out_points" in bev_with_new_label:
                cv2.imwrite(bev_path_filename + "original.png", image_02_image)
                write_ply(bev_path_filename + ".ply", bev_with_new_label['save_out_points'],
                          bev_with_new_label['save_out_colors'])

            if "debug" in bev_with_new_label:
                debug_values = bev_with_new_label.copy()

                # delete images from the json - these are not json serializable
                debug_values.pop('data')
                debug_values.pop('generated_osm')
                debug_values.pop('negative_osm')

                json_path_filename = bev_path_filename + ".json"
                with open(json_path_filename, 'w') as outfile:
                    json.dump(debug_values, outfile)
                bev_with_new_label['json_path_filename'] = json_path_filename

                # for debuggin' purposes
                if "save_out_points" in bev_with_new_label:
                    debug_values.pop('save_out_points')
                    debug_values.pop('save_out_colors')

        return bev_with_new_label

    def __filterdistance(self, distance):
        aanet = []
        alvaromask = []
        image_02 = []

        for aafile, mask, image in zip(self.aanet, self.alvaromask, self.image_02):
            head, filename = os.path.split(image)
            head, _ = os.path.split(os.path.normpath(head))
            datapath = os.path.join(head, 'frames_topology.txt')
            name, _ = os.path.splitext(filename)
            gtdata = pd.read_csv(datapath, sep=';', header=None, dtype=str)
            if float(gtdata.loc[gtdata[0] == name][1]) < distance:
                image_02.append(image)
                alvaromask.append(mask)
                aanet.append(aafile)

        self.aanet = aanet
        self.alvaromask = alvaromask
        self.image_02 = image_02


class fromGeneratedDataset(Dataset):

    def __init__(self, folders, distance, transform=None,
                 rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True, canonical=False, addGeneratedOSM=True,
                 decimateStep=1, savelist=False, loadlist=False, random_rate=1.0):
        # TODO addGeneratedOSM should not be the default behavior; set FALSE here anc call with TRUE as needed

        """
        Args:

            # THE FOLLOWING PARAMETERS ARE COPIED FROM <teacher_tripletloss_generated>
            rnd_width: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)
            rnd_angle: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)
            rnd_spatial: parameter for uniform spatial cross position (center of the crossing area)
            canonical: set false to avoid generating the "canonical" crossings
            noise: whether to add or not noise in the image (pixel level)
            decimateStep: use this value to decimate the 100-elements of the generated-dataset; set as "::STEP"

            addGeneratedOSM: whether to add the generated OSM intersection (to train as student)

            savelist: saves ALL full file names (path+file); this is ONE for ALL the dataset; todo: fix for k-fold!
            loadlist: load the previous ones.. once you have saved them. consider using a script like
                      "standardization.py" , or other code, that load this dataloader just once, so the "saved/npz" will
                      be created
        """

        if not isinstance(decimateStep, int) and decimateStep > 0:
            print("decimateStep must be an integer > 0. Passed: ", decimateStep)
            exit(-1)

        if savelist and loadlist:
            print("load and save at the same time")
            exit(-1)

        self.transform_generated = transform

        self.bev_images = []
        self.bev_labels = []

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.canonical = canonical
        self.addGeneratedOSM = addGeneratedOSM
        self.random_rate = random_rate

        tic = time.time()

        fullfilename = os.path.join(os.path.commonpath(folders.tolist()),
                                    "fromGeneratedDataset" + str(distance) + ".npz")

        if loadlist and os.path.isfile(fullfilename):

            print("\nLoading existing file list from " + fullfilename)

            self.bev_images = np.load(fullfilename)['bev_images'].tolist()
            self.bev_labels = np.load(fullfilename)['bev_labels'].tolist()

        else:

            if os.path.isfile(fullfilename):
                print("A saved version of the datasets exists. Consider using --loadlist")

            for folder in folders:
                if os.path.isdir(os.path.join(folder, 'image_02')):
                    folder = os.path.join(folder, 'image_02')
                    current_filelist = glob.glob1(str(folder), "*.png")
                else:
                    current_filelist = glob.glob1(str(folder), "*.png")
                for file in current_filelist:
                    bev_filename = os.path.join(folder, file)
                    json_filename = str(bev_filename.replace('png', 'png.json'))

                    assert os.path.exists(bev_filename), "no bev file"
                    assert os.path.exists(json_filename), "no json file"

                    self.bev_images.append(bev_filename)

                    with open(json_filename) as json_file:
                        bev_label = json.load(json_file)
                        self.bev_labels.append(bev_label['label'])

            self.__filterdistance(distance)

        if savelist:
            np.savez_compressed(fullfilename,
                                bev_images=np.asarray(self.bev_images),
                                bev_labels=np.asarray(self.bev_labels))

            print("File list saved to " + fullfilename)

        # decimate
        print("The dataset will be decimated taking 1 out of " + str(decimateStep) + " elements")
        self.bev_images = self.bev_images[::decimateStep]
        self.bev_labels = self.bev_labels[::decimateStep]

        self.feasible_intersections = list(set(self.bev_labels))

        toc = time.time()
        print("[fromGeneratedDataset] - " + str(len(self.bev_labels)) + " elements loaded in " + str(
            time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

    def __len__(self):

        return len(self.bev_labels)

    def __getitem__(self, idx):

        assert os.path.isfile(self.bev_images[idx]), "os.path.isfile(self.bev_images[idx]) -- no image"

        bev_image = Image.open(self.bev_images[idx])  # pil image open is faster than opencv
        bev_label = self.bev_labels[idx]

        # Sample an intersection given a label; this is used in the STUDENT training
        r = [0, 1, 2, 3, 4, 5, 6]
        r.remove(bev_label)
        negative_label = random.choice(r)

        if self.addGeneratedOSM:
            # This can be used for loss functions without the centroids
            generated_osm = test_crossing_pose(crossing_type=bev_label, save=False, rnd_width=self.rnd_width,
                                               rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                               sampling=not self.canonical, random_rate=self.random_rate)
            generated_osm_negative = test_crossing_pose(crossing_type=negative_label, save=False,
                                                        rnd_width=self.rnd_width,
                                                        rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial,
                                                        noise=self.noise,
                                                        sampling=not self.canonical, random_rate=self.random_rate)
            sample = {'data': bev_image,
                      'label': bev_label,
                      'neg_label': negative_label,
                      'image_path': self.bev_images[idx],
                      'generated_osm': generated_osm[0],
                      'negative_osm': generated_osm_negative[0]}
        else:
            # this can be used for loss functions with centroids
            sample = {'data': bev_image,
                      'label': bev_label,
                      'neg_label': negative_label,
                      'image_path': self.bev_images[idx]
                      }

        if self.transform_generated:
            sample['data'] = self.transform_generated(sample['data'])

        return sample

    def __filterdistance(self, distance):
        images = []
        labels = []
        datapath_last = ""
        for file, label in zip(self.bev_images, self.bev_labels):
            head, filename = os.path.split(file)
            head, folder = os.path.split(head)
            if not os.path.isfile(os.path.join(head, 'frames_topology.txt')):
                head, seq = os.path.split(head)
            if not os.path.isfile(os.path.join(head, 'frames_topology.txt')):
                head = os.path.join(os.path.join(head, 'data_raw'), folder)
            datapath = os.path.join(head, 'frames_topology.txt')
            name, _ = os.path.splitext(filename)
            name = name.split('.')[0]
            if datapath != datapath_last:
                gtdata = pd.read_csv(datapath, sep=';', header=None, dtype=str)
                datapath_last = datapath
            panda_to_numpy = np.asarray(gtdata)
            at = np.where(panda_to_numpy[:, 0] == np.asarray([name]))[0][0]
            if float(panda_to_numpy[at, 1]) < distance:
                images.append(file)
                labels.append(label)
        self.bev_images = images
        self.bev_labels = labels


class teacher_tripletloss(Dataset):

    def __init__(self, folders, distance, include_insidecrossing=False, transform=None, noise=True, canonical=True,
                 random_rate=1.0):
        """

        This dataloader uses "REAL" intersection, using the OSM data; this data is pre-generated from the OSM and the
        old software from ICRA 2019 by Cattaneo/Ballardini.

        Args:
            folders:    all the folders, like
                        /home/malvaro/Documentos/DualBiSeNet/data_raw
                                ├── mkdir 2011_09_30_drive_0018_sync
                                ├── mkdir 2011_09_30_drive_0020_sync
                                ├── mkdir 2011_09_30_drive_0027_sync
                                ├── mkdir 2011_09_30_drive_0028_sync
                                ├── mkdir 2011_09_30_drive_0033_sync
                                ├── mkdir 2011_09_30_drive_0034_sync
                                ├── mkdir 2011_10_03_drive_0027_sync
                                └── mkdir 2011_10_03_drive_0034_sync

            distance:   distance to consider to add intersection images; only images withing this value will be included.
                        This value was set to 20 in past ICRA works (with KITTI)

            transform:  transforms to the image

            include_insidecrossing: whether include or not the frames in which the vehicle is almost inside the crossing
                                    by the definition of our dataset

            noise: if set, we'll add noise to the OSM-OG in a same way we've done in teacher_tripletloss_generated.

            canonical: set false to avoid generating the "canonical" crossings

        """

        self.noise = noise
        self.canonical = canonical
        self.random_rate = random_rate
        self.transform = transform

        # osm_files = []
        # osm_types = []
        # osm_distances = []

        osm_data = []

        try:
            tic = time.time()
            # create the list of elements in all folders
            for folder in folders:
                folder_osm = os.path.join(folder, 'OSM')
                folder_img = os.path.join(folder, 'image_02')
                folder_oxts = os.path.join(folder, 'oxts/data')

                # check if the directory exists
                if os.path.isdir(folder_osm) and os.path.isdir(folder_img) and os.path.isdir(folder_oxts):

                    gt_path = os.path.join(folder, 'frames_topology.txt')

                    # check if the file exists
                    if os.path.isfile(gt_path):

                        # read data
                        gt_data = pd.read_csv(gt_path, sep=';', header=None, dtype=str)

                        for file in sorted(os.listdir(folder_osm)):

                            # check if is a PNG .. remember the files should be like 0000000000.png [10digits].png
                            if file.endswith('.png'):

                                index = os.path.splitext(file)[0]

                                # little check; ensure the "index" is a number
                                assert index.isdigit()

                                osm_data_distance = float(gt_data.loc[gt_data[0] == index][1])
                                osm_data_type = int(gt_data.loc[gt_data[0] == index][2])
                                osm_data_insidecrossing = int(gt_data.loc[gt_data[0] == index][3])

                                oxts_file = os.path.join(folder_oxts, str.replace(file, "png", "txt"))
                                oxts_data = pd.read_csv(oxts_file, sep=' ', header=None, dtype=str)
                                oxts_lat = oxts_data[0][0]
                                oxts_lon = oxts_data[1][0]

                                # check whether the correspondent CAMERA IMAGE exists
                                # assert os.path.isfile(os.path.join(folder_img, file))

                                if include_insidecrossing or osm_data_insidecrossing == 0:

                                    if osm_data_distance < distance:
                                        osm_data.append(
                                            [os.path.join(folder, "OSM", file), osm_data_distance, osm_data_type,
                                             osm_data_insidecrossing, oxts_lat, oxts_lon])

                                        osm_data.append([os.path.join(folder, "OSM", file),
                                                         osm_data_distance,
                                                         osm_data_type,
                                                         osm_data_insidecrossing,
                                                         oxts_lat,
                                                         oxts_lon])

                                        # osm_files.append(os.path.join(folder, "OSM", file))
                                        # osm_types.append(osm_data_type)
                                        # osm_distances.append(osm_data_distance)

            toc = time.time()
            print("[teacher_tripletloss] - " + str(len(osm_data)) + " elements loaded in " +
                  str(time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

        except Exception as e:
            if isinstance(e, SystemExit):
                exit()
            print(e)
            exit()

        self.osm_data = osm_data

    def __len__(self):
        return len(self.osm_data)

    def __getitem__(self, idx):

        """
        0 filename
        1 meters
        2 typology
        3 in crossing

        """

        # identify the typology for anchor item
        anchor_type = self.osm_data[idx][2]

        # create positive and negative list based on the anchor item typology
        positive_list = [element for element in self.osm_data if element[2] == anchor_type]
        negative_list = [element for element in self.osm_data if element[2] != anchor_type]

        # remove the anchor item from the positive list
        positive_list.remove(self.osm_data[idx])
        positive_item = random.choice(positive_list)
        negative_item = random.choice(negative_list)

        anchor_image = cv2.imread(self.osm_data[idx][0], cv2.IMREAD_UNCHANGED)
        positive_image = cv2.imread(positive_item[0], cv2.IMREAD_UNCHANGED)
        negative_image = cv2.imread(negative_item[0], cv2.IMREAD_UNCHANGED)
        if self.canonical:  # set canonical to False to speedup this dataloader
            canonical_image = test_crossing_pose(crossing_type=anchor_type, save=False, noise=self.noise,
                                                 sampling=False, random_rate=self.random_rate)[0]
        else:
            canonical_image = [0]

        ground_truth_img = cv2.imread(
            os.path.dirname(str(self.osm_data[idx][0])) + "_TYPES/" + str(self.osm_data[idx][2]) + ".png",
            cv2.IMREAD_COLOR)

        # adding noise
        if self.noise:
            # TODO change the default value to uniform and stop doing this shit; do something also for
            #      elements_multiplier, it's current default value is 1.0 but we always used 3.0!
            anchor_image = Crossing.add_noise(self, anchor_image, elements_multiplier=3.0, distribution="uniform")
            positive_image = Crossing.add_noise(self, positive_image, elements_multiplier=3.0, distribution="uniform")
            negative_image = Crossing.add_noise(self, negative_image, elements_multiplier=3.0, distribution="uniform")

        # a = plt.figure()
        # plt.imshow(positive_image)
        # send_telegram_picture(a, "positive_image")
        # plt.close('all')

        sample = {'anchor': anchor_image,
                  'positive': positive_image,
                  'negative': negative_image,
                  'canonical': canonical_image,
                  'label_anchor': anchor_type, 'label_positive': positive_item[2],  # [2] is the type
                  'label_negative': negative_item[2],  # [2] is the type
                  'filename_anchor': self.osm_data[idx][0],  # [0] is the filename
                  'filename_positive': positive_item[0], 'filename_negative': negative_item[0],
                  'ground_truth_image': ground_truth_img,  # for debugging purposes
                  'anchor_oxts_lat': self.osm_data[idx][4],  # [4] lat
                  'anchor_oxts_lon': self.osm_data[idx][5],  # [5] lon
                  'positive_oxts_lat': positive_item[4],  # [4] lat
                  'positive_oxts_lon': positive_item[5],  # [5] lon
                  'negative_oxts_lat': negative_item[4],  # [4] lat
                  'negative_oxts_lon': negative_item[5]  # [5] lon
                  }

        if self.transform:
            sample['anchor'] = self.transform(sample['anchor'])
            sample['positive'] = self.transform(sample['positive'])
            sample['negative'] = self.transform(sample['negative'])

        if self.canonical:
            sample['canonical'] = self.transform(sample['canonical'])

        return sample


class teacher_tripletloss_generated(Dataset):

    def __init__(self, elements=1000, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True, canonical=True,
                 transform=None, random_rate=1.0, crossing_type_set=None):
        """

        This dataloader uses "RUNTIME-GENERATED" intersections (this differs from teacher_tripletloss dataloader that
        uses the OSM data).

        Args:

            elements:   how many elements do you want in this "generator"

            rnd_width: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)
            rnd_angle: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)
            rnd_spatial: parameter for uniform spatial cross position (center of the crossing area)
            canonical: set false to avoid generating the "canonical" crossings
            noise: whether to add or not noise in the image (pixel level)
            random_rate: this parameter multiplies all the rnd_xxxxx values; used to create such as "learning rate"

            transform:  transforms to the image
            crossing_type_set: if set, this defines the types of available crossing type in this dataset. pass a LIST

        """

        if crossing_type_set is None:
            crossing_type_set = [0, 1, 2, 3, 4, 5, 6]

        self.elements = elements
        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.canonical = canonical
        self.random_rate = random_rate

        self.transform_triplet = transform

        # Generate a list of crossings; will be then used during the sampling process; is just a list of int s
        osm_types = []

        for crossing_type in crossing_type_set:
            for _ in range(0, elements):
                # sample = test_crossing_pose(crossing_type=crossing_type, noise=True, rnd_width=1.0,
                # save=False, random_rate = )
                osm_types.append(crossing_type)

        self.samples_triplet = osm_types

    def __len__(self):
        return len(self.samples_triplet)

    def set_rnd_angle(self, rnd_angle):
        """

        Args:
            rnd_angle: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)

        Returns: the new self variable value

        """
        self.rnd_angle = rnd_angle
        return self.rnd_angle

    def set_rnd_width(self, rnd_width):
        """

        Args:
            rnd_width: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)

        Returns: the new self variable value

        """
        self.rnd_width = rnd_width
        return self.rnd_width

    def set_rnd_spatial(self, rnd_spatial):
        """

        Args:
            rnd_spatial: parameter for uniform spatial cross position (center of the crossing area)

        Returns: the new self variable value

        """
        self.rnd_spatial = rnd_spatial
        return self.rnd_spatial

    def set_random_rate(self, random_rate):
        """

        Args:
            random_rate: this parameter multiplies all the rnd_xxxxx values; used to create such as "learning rate"
                         typically used through the epochs

        Returns: the new self variable value

        """
        self.random_rate = random_rate
        return self.random_rate

    def get_rnd_angle(self):

        """

        Returns: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)

        """
        return self.rnd_angle

    def get_rnd_width(self):
        """

        Returns: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)

        """

        return self.rnd_width

    def get_rnd_spatial(self):
        """

        Returns: parameter for uniform spatial cross position (center of the crossing area)

        """
        return self.rnd_spatial

    def get_random_rate(self):
        """

        Returns: parameter for random_rate

        """

        return self.random_rate

    def __getitem__(self, idx):

        """
        0 filename
        1 meters
        2 typology
        3 in crossing

        """

        # safe to delete, this is for the SEED test
        # if idx == 0:
        #     print("Random seed to check: " + str(np.random.rand() ))

        # identify the typology for anchor item
        anchor_type = self.samples_triplet[idx]

        # create positive and negative list based on the anchor item typology
        positive_list = [element for element in self.samples_triplet if element == anchor_type]
        negative_list = [element for element in self.samples_triplet if element != anchor_type]

        # remove the anchor item from the positive list
        positive_list.remove(self.samples_triplet[idx])
        positive_item = random.choice(positive_list)
        negative_item = random.choice(negative_list)

        anchor_image = test_crossing_pose(crossing_type=anchor_type, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          random_rate=self.random_rate)
        positive_image = test_crossing_pose(crossing_type=positive_item, save=False, rnd_width=self.rnd_width,
                                            rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                            random_rate=self.random_rate)
        negative_image = test_crossing_pose(crossing_type=negative_item, save=False, rnd_width=self.rnd_width,
                                            rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                            random_rate=self.random_rate)
        if self.canonical:  # set canonical to False to speedup this dataloader
            canonical_image = test_crossing_pose(crossing_type=anchor_type, save=False, noise=self.noise,
                                                 sampling=False, random_rate=self.random_rate)
        else:
            canonical_image = [0]

        # anchor_image = cv2.imread(self.samples[idx][0], cv2.IMREAD_UNCHANGED)

        sample = {'anchor': anchor_image[0],  # [0] is the image
                  'positive': positive_image[0],
                  'negative': negative_image[0],
                  'canonical': canonical_image[0],
                  'label_anchor': anchor_type,
                  'label_positive': positive_item,  # [0] is the type
                  'label_negative': negative_item,  # [0] is the type
                  'ground_truth_image': anchor_image[0],  # for debugging purposes | in this dataloader is = the anchor
                  'anchor_xx': anchor_image[1],  # [1] is the xx coordinate
                  'anchor_yy': anchor_image[2],  # [2] is the yy coordinate
                  'positive_xx': positive_image[1],
                  'positive_yy': positive_image[2],
                  'negative_xx': negative_image[1],
                  'negative_yy': negative_image[2],

                  # the following are not used; are here to maintain the compatibility with "teacher_tripletloss"
                  'filename_anchor': 0, 'filename_positive': 0, 'filename_negative': 0, 'anchor_oxts_lat': 0,
                  'anchor_oxts_lon': 0, 'positive_oxts_lat': 0, 'positive_oxts_lon': 0, 'negative_oxts_lat': 0,
                  'negative_oxts_lon': 0}

        if self.transform_triplet:
            sample['anchor'] = self.transform_triplet(sample['anchor'])
            sample['positive'] = self.transform_triplet(sample['positive'])
            sample['negative'] = self.transform_triplet(sample['negative'])

        return sample


class triplet_OBB(teacher_tripletloss_generated, fromGeneratedDataset, Dataset):
    """
        fromGeneratedDataset: no sense to decimate this dataset, the speedup is achieved decreasing
                              teacher_tripletloss_generated
    """

    def __init__(self, folders, distance, elements=1000, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                 canonical=True, transform_osm=None, transform_bev=None, random_rate=1.0, loadlist=True, savelist=False,
                 ):

        fromGeneratedDataset.__init__(self, folders, distance, transform=transform_bev, rnd_width=rnd_width,
                                      rnd_angle=rnd_angle, rnd_spatial=rnd_spatial, noise=noise, canonical=canonical,
                                      addGeneratedOSM=True, savelist=savelist, loadlist=loadlist, decimateStep=1)

        teacher_tripletloss_generated.__init__(self, elements=elements, rnd_width=rnd_width, rnd_angle=rnd_angle,
                                               rnd_spatial=rnd_spatial, noise=noise, canonical=canonical,
                                               transform=transform_osm, random_rate=random_rate,
                                               crossing_type_set=self.feasible_intersections)

        self.transform_osm = transform_osm
        self.transform_bev = transform_bev

    def __len__(self):
        # In this multi inherit class we have both [teacher_tripletloss_generated] and [fromGeneratedDataset] items.
        # in OBB , OSM + BEV + BEV we want that our list of elements is like the teacher_tripletloss_generated one.
        # in BOO , BEV + OSM + OSM we want that our list of elements is like the fromGeneratedDataset
        return len(self.samples_triplet)

    def __getitem__(self, idx):
        # OBB ... so to get the OSM we can simply get the anchor from the teacher_tripletloss_generated triplet
        sample_ttg = teacher_tripletloss_generated.__getitem__(self, idx)

        OSM = sample_ttg['anchor']

        # for the POSITIVE and NEGATIVE, we can extract the indices from the self.bev_labels using the OSM anchor label
        positive_list = [idx for idx, element in enumerate(self.bev_labels) if element == sample_ttg['label_anchor']]
        negative_list = [idx for idx, element in enumerate(self.bev_labels) if element != sample_ttg['label_anchor']]

        try:
            positive_item = random.choice(positive_list)
            negative_item = random.choice(negative_list)
        except:
            print("lenght of positive_list: ", len(positive_list))
            print("lenght of negative_list: ", len(negative_list))
            print("index was: ", idx)
            print("sample_ttg['label_anchor']: ", sample_ttg['label_anchor'])
            exit(-1)

        # once you have the items, simply call teh getitem of fromGeneratedDataset! this will return a sample.
        SAMPLE_POSITIVE = fromGeneratedDataset.__getitem__(self, positive_item)
        SAMPLE_NEGATIVE = fromGeneratedDataset.__getitem__(self, negative_item)

        # get the data!
        BEV_POSITIVE = SAMPLE_POSITIVE['data']
        BEV_NEGATIVE = SAMPLE_NEGATIVE['data']

        sample = {'OSM_anchor': OSM,
                  'BEV_positive': BEV_POSITIVE,
                  'BEV_negative': BEV_NEGATIVE,
                  'label_anchor': sample_ttg['label_anchor'],
                  'label_positive': SAMPLE_POSITIVE['label'],
                  'label_negative': SAMPLE_NEGATIVE['label'],
                  'filename_positive': SAMPLE_POSITIVE['image_path'],
                  'filename_negative': SAMPLE_NEGATIVE['image_path']}

        if self.transform_osm:
            sample['OSM_anchor'] = self.transform_osm(sample['OSM_anchor'])

        if self.transform_bev:
            sample['BEV_positive'] = self.transform_bev(sample['BEV_positive'])
            sample['BEV_negative'] = self.transform_bev(sample['BEV_negative'])

        # DEBUG -- send in telegram. Little HACK for the ANCHOR, get a sub-part of image ...
        '''
        emptyspace = 255 * torch.ones([224, 30, 3], dtype=torch.uint8)
        a = plt.figure()
        plt.imshow(torch.cat((torch.tensor(sample['OSM_anchor'])[38:262, 38:262, :], emptyspace,
                              torch.tensor(sample['BEV_positive']), emptyspace, torch.tensor(sample['BEV_negative'])),
                             1))
        send_telegram_picture(a, "OSM | BEV(positive) | BEV(negative)" +
                              "\nWarning! Image of OSM_anchor is CROPPED__\n" +
                              "\nlabel_anchor: " + str(sample['label_anchor']) +
                              "\nlabel_positive: " + str(sample['label_positive']) +
                              "\nlabel_negative: " + str(sample['label_negative']) +
                              "\nfilename positive: " + str(sample['filename_positive']) +
                              "\nfilename negative: " + str(sample['filename_negative'])
                              )
        '''
        return sample


class triplet_BOO(fromGeneratedDataset, Dataset):
    """
        In this dataloader, <<teacher_tripletloss_generated>> is not actually used; what we really need is the
        <<test_crossing_pose>> that is used inside <<teacher_tripletloss_generated>> but using the class to get
        the OSM-pos/neg seemed a little awkward and required changes in that class; for this reason we directly
        used the <<test_crossing_pose>> here.

        teacher_tripletloss_generated is here just to keep this triplet_BOO with same init as triplet_BOO :)

    """

    def __init__(self, folders, distance, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                 canonical=True, transform_osm=None, transform_bev=None, random_rate=1.0, loadlist=False,
                 savelist=False,
                 decimateStep=1):
        fromGeneratedDataset.__init__(self, folders, distance, transform=transform_bev, rnd_width=rnd_width,
                                      rnd_angle=rnd_angle, rnd_spatial=rnd_spatial, noise=noise, canonical=canonical,
                                      addGeneratedOSM=False, decimateStep=decimateStep, savelist=savelist,
                                      loadlist=loadlist)

        self.types = [0, 1, 2, 3, 4, 5, 6]

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.random_rate = random_rate
        self.canonical = canonical
        self.transform_osm = transform_osm
        self.transform_bev = transform_bev

    def __len__(self):
        # In this multi inherit class we have both [teacher_tripletloss_generated] and [fromGeneratedDataset] items.
        # in OBB , OSM + BEV + BEV we want that our list of elements is like the teacher_tripletloss_generated one.
        # in BOO , BEV + OSM + OSM we want that our list of elements is like the fromGeneratedDataset
        return len(self.bev_labels)

    def __getitem__(self, idx):
        # BOO ... get a random BEV from fromGeneratedDataset
        sample_fgd = fromGeneratedDataset.__getitem__(self, idx)

        BEV = sample_fgd['data']
        item_positive = sample_fgd['label']
        item_negative = random.choice([element for element in self.types if element != item_positive])

        OSM_positive = test_crossing_pose(crossing_type=item_positive, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=self.random_rate)
        OSM_negative = test_crossing_pose(crossing_type=item_negative, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=self.random_rate)

        sample = {'anchor': BEV,
                  'positive': OSM_positive[0],
                  'negative': OSM_negative[0],
                  'label_anchor': sample_fgd['label'],
                  'label_positive': item_positive,
                  'label_negative': item_negative,
                  'filename_anchor': sample_fgd['image_path']
                  }
        if self.transform_osm:
            sample['positive'] = self.transform_osm(sample['positive'])
            sample['negative'] = self.transform_osm(sample['negative'])

        # DEBUG -- send in telegram. Little HACK for the ANCHOR, get a sub-part of image ...
        # emptyspace = 255 * torch.ones([224, 30, 3], dtype=torch.uint8)
        # a = plt.figure()
        # plt.imshow(torch.cat((torch.tensor(sample['BEV_anchor']), emptyspace,
        # torch.tensor(sample['OSM_positive'])[38:262, 38:262, :], emptyspace,
        # torch.tensor(sample['OSM_negative'])[38:262, 38:262, :]), 1))
        # send_telegram_picture(a, \
        # "OSM | BEV(positive) | BEV(negative)" + \
        # "\nWarning: images of OSM_positive and OSM_negative are CROPPED!\n" + \
        # "\nlabel_anchor: " + str(sample['label_anchor']) + \
        # "\nlabel_positive: " + str(sample['label_positive']) + \
        # "\nlabel_negative: " + str(sample['label_negative']) + \
        # "\nfilename positive: " + str(sample['filename_anchor']) \
        # )

        return sample


class triplet_ROO(Kitti2011_RGB, Dataset):
    """
        In this dataloader, <<teacher_tripletloss_generated>> is not actually used; what we really need is the
        <<test_crossing_pose>> that is used inside <<teacher_tripletloss_generated>> but using the class to get
        the OSM-pos/neg seemed a little awkward and required changes in that class; for this reason we directly
        used the <<test_crossing_pose>> here.

        teacher_tripletloss_generated is here just to keep this triplet_BOO with same init as triplet_BOO :)

    """

    def __init__(self, folders, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                 canonical=False, transform_osm=None, transform_rgb=None, random_rate=1.0,
                 ):
        Kitti2011_RGB.__init__(self, folders, transform=transform_rgb)

        self.types = [0, 1, 2, 3, 4, 5, 6]

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.random_rate = random_rate
        self.canonical = canonical
        self.transform_osm = transform_osm

    def __len__(self):
        # In this multi inherit class we have both [teacher_tripletloss_generated] and [Kitti2011_RGB] items.
        # in ROO , RGB + OSM + OSM we want that our list of elements is like the Kitti2011_RGB
        return len(self.image_02)

    def __getitem__(self, idx):
        # ROO ... get a random RGB//Homography from Kitti2011_RGB
        sample_rgb = Kitti2011_RGB.__getitem__(self, idx)

        rgb_image = sample_rgb['data']
        label_positive = sample_rgb['label']
        label_negative = random.choice([element for element in self.types if element != label_positive])

        OSM_positive = test_crossing_pose(crossing_type=label_positive, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=self.random_rate)
        OSM_negative = test_crossing_pose(crossing_type=label_negative, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=self.random_rate)

        sample = {'anchor': rgb_image,
                  'positive': OSM_positive[0],
                  'negative': OSM_negative[0],
                  'label_anchor': label_positive,
                  'label_positive': label_positive,
                  'label_negative': label_negative,
                  }
        if self.transform_osm:
            sample['positive'] = self.transform_osm(sample['positive'])
            sample['negative'] = self.transform_osm(sample['negative'])

        return sample


class triplet_ROO_360(kitti360, Dataset):
    """
        In this dataloader, <<teacher_tripletloss_generated>> is not actually used; what we really need is the
        <<test_crossing_pose>> that is used inside <<teacher_tripletloss_generated>> but using the class to get
        the OSM-pos/neg seemed a little awkward and required changes in that class; for this reason we directly
        used the <<test_crossing_pose>> here.

        teacher_tripletloss_generated is here just to keep this triplet_BOO with same init as triplet_BOO :)

    """

    def __init__(self, path, sequences, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                 canonical=False, transform_osm=None, transform_rgb=None, transform_3d=None):

        if transform_rgb:
            kitti360.__init__(self, path, sequences, transform=transform_rgb)
        elif transform_3d:
            kitti360.__init__(self, path, sequences, transform=transform_3d)

        self.types = [0, 1, 2, 3, 4, 5, 6]

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.canonical = canonical
        self.transform_osm = transform_osm
        self.transform_rgb = transform_rgb
        self.transform_3d = transform_3d

    def __len__(self):
        # In this multi inherit class we have both [teacher_tripletloss_generated] and [kitti360] items.
        # in ROO , RGB + OSM + OSM we want that our list of elements is like the Kitti2011_RGB
        return len(self.images)

    def __getitem__(self, idx):
        # ROO ... get a random RGB//Homography//3D from kitti360
        sample_rgb = kitti360.__getitem__(self, idx)

        rgb_image = sample_rgb['data']
        label_positive = sample_rgb['label']
        label_negative = sample_rgb['neg_label']

        OSM_positive = test_crossing_pose(crossing_type=label_positive, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=1.0)
        OSM_negative = test_crossing_pose(crossing_type=label_negative, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise,
                                          sampling=not self.canonical, random_rate=1.0)

        sample = {'anchor': rgb_image,
                  'positive': OSM_positive[0],
                  'negative': OSM_negative[0],
                  'label_anchor': label_positive,
                  'label_positive': label_positive,
                  'label_negative': label_negative,
                  }
        if self.transform_osm:
            sample['positive'] = self.transform_osm(sample['positive'])
            sample['negative'] = self.transform_osm(sample['negative'])

        return sample


class fromAANETandDualBisenet360(Dataset):

    def __init__(self, folders, distance=0, transform=None):
        """
        Inspired by fromAANETandDualBisenet, but with KITTI360 dataset.

        no 'filterdistance' is here since we still don't have the GPS positions, so is "up to my eyes"

        Args:
            folders: location of the imagery
            distance: UNUSED


        """

        self.transform = transform

        aanet = []
        image_02 = []
        classification = []

        for folder in folders:
            folder_aanet = os.path.join(folder, 'pred')
            folder_image_02 = os.path.join(folder, 'left')
            for file in sorted(os.listdir(folder_aanet)):
                image_02_file = file.replace("_pred.npz", ".png")

                if os.path.isfile(os.path.join(folder_aanet, file)) and os.path.isfile(
                        os.path.join(folder_image_02, image_02_file)):
                    aanet.append(os.path.join(folder_aanet, file))
                    image_02.append(os.path.join(folder_image_02, image_02_file))
                    if os.path.split(folder)[1].isnumeric():
                        classification.append(int(os.path.split(folder)[1]))
                else:
                    print("Loader error")
                    print(os.path.join(folder_aanet, file))
                    print(os.path.join(folder_image_02, image_02_file))

        self.aanet = aanet
        self.image_02 = image_02
        self.classification = classification

        assert len(self.aanet) > 0, 'Training files missing [aanet]'
        assert len(self.image_02) > 0, 'Training files missing [imagery]'
        assert len(self.classification) > 0, 'error on classification'

    def __len__(self):

        return len(self.aanet)

    def __getitem__(self, idx):

        # Select file subset
        aanet_file = self.aanet[idx]
        image_02_file = self.image_02[idx]

        dict_data = load(aanet_file)
        aanet_image = dict_data['arr_0']
        image_02_image = cv2.imread(image_02_file, cv2.IMREAD_UNCHANGED)

        # Obtaining ground truth
        gTruth = self.classification[idx]

        sample = {'aanet': aanet_image,
                  'alvaromask': None,  # kept for compatibilty
                  'image_02': image_02_image,
                  'label': gTruth}

        assert self.transform, "no transform list provided"

        # call all the transforms
        bev_with_new_label = self.transform(sample)

        # check whether the <GenerateNewDataset> transform was included (check the path); if yes, save the generated BEV
        # please notice that the path.parameter isn't actually used, was our first intention, but then was simpler to do
        # the following... OPTIMIZE change "path" as string with boolean
        if "path" in bev_with_new_label:
            # folder, file = os.path.split(image_02_file.replace("data_raw", "data_raw_bev").replace("image_02", ""))
            # dataset_path = os.path.join(bev_with_new_label['path'], os.path.basename(folder))
            # base_file_star = str(file.split(".")[0]) + "*"
            # last_number = len([x for x in current_filelist if "json" not in x])
            # final_filename = str(file.split(".")[0]) + '.' + str(last_number + 1).zfill(3) + ".png"

            dataset_path = os.path.join(bev_with_new_label['path'], str(bev_with_new_label['label']))
            base_file_star = os.path.splitext(os.path.split(image_02_file)[1])[0]
            current_filelist = glob.glob1(dataset_path, base_file_star + '*')
            last_number = len([x for x in current_filelist if "json" not in x])
            final_filename = base_file_star + '.' + str(last_number + 1).zfill(3) + '.png'
            bev_path_filename = os.path.join(dataset_path, final_filename)

            # path must already exist!
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            flag = cv2.imwrite(bev_path_filename, bev_with_new_label['data'])
            assert flag, "can't write file"
            bev_with_new_label['bev_path_filename'] = bev_path_filename

            # for debuggin' purposes
            if "save_out_points" in bev_with_new_label:
                cv2.imwrite(bev_path_filename + "original.png", image_02_image)
                write_ply(bev_path_filename + ".ply", bev_with_new_label['save_out_points'],
                          bev_with_new_label['save_out_colors'])

            if "debug" in bev_with_new_label:
                debug_values = bev_with_new_label.copy()

                # delete images from the json - these are not json serializable
                debug_values.pop('data')
                debug_values.pop('generated_osm')
                debug_values.pop('negative_osm')

                json_path_filename = bev_path_filename + ".json"
                with open(json_path_filename, 'w') as outfile:
                    json.dump(debug_values, outfile)
                bev_with_new_label['json_path_filename'] = json_path_filename

                # for debuggin' purposes
                if "save_out_points" in bev_with_new_label:
                    debug_values.pop('save_out_points')
                    debug_values.pop('save_out_colors')

        return bev_with_new_label


class lstm_txt_dataloader(txt_dataloader, Dataset):
    """

    This dataloader is intended to be used with just a "filename" passed, containing the list of
    images to be used. No OS walk will be used. Is intended to be used as the dataloader in
    class alcala26012021(Dataset), but for sequences.

    Do not copy, use inheritance!

    Used like this:

    dataset = Sequences_alcala26012021_Dataloader(
                                path_filename='/home/ballardini/Desktop/alcala-26.01.2021/train_list.txt',
                                usePIL=False)

    """

    def __init__(self, path_filename=None, transform=None, usePIL=True, isSequence=True, all_in_ram=False,
                 fixed_lenght=0, verbose=True):
        """

                THIS IS THE DATALOADER USES the split files generated with labelling-script.py

                Args:
                    path_filename (string): filename with all the files that you want to use; this dataloader uses a file
                    with all the images, does not walk a os folder!
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                    usePIL: default True, but if not, return numpy-arrays!

                    isSequence : this parameter specifies that this dataloader is using sequences! used together with
                                 the abstract class

                    fixed_lenght: if 0 -> use all sequence
                                  if 1 -> use the last 'min_elements' of the sequence
                                  if 2 -> use 'equal-spaced' sequence (linear space)

        """

        # call the super init class. from this we'll have
        # self.transform = transform
        # self.images = trainimages
        # self.labels = trainlabels
        # self.usePIL = usePIL
        # Sequences_alcala26012021_Dataloader.__init__(self, path_filename_list, transform, usePIL)
        super().__init__(path_filename, transform, usePIL, isSequence=isSequence, verbose=verbose)

        sequences = {}
        last_seq = 0
        sequences, last_seq, min_elements = self.__get_sequences(self, '', self.images, last_seq, sequences)

        self.sequences = sequences
        self.all_in_ram = all_in_ram
        self.images_in_ram = {}
        self.min_elements = min_elements
        self.fixed_lenght = fixed_lenght

        # workaround to open lot of files
        # https://github.com/python-pillow/Pillow/issues/1237
        if self.all_in_ram:
            for img_filename in self.images:
                img_ = Image.open(img_filename)
                self.images_in_ram[img_filename] = img_.copy()
                img_.close()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """

        Args:
            idx: index of the sequence. this is generated by the dataloader

        Returns:

            the sample containing the image list and the label of the sequence. if not all the frames have the same
            label, an "exit" will be called.

        """

        # print('Class: ' + __class__.__name__ + " -- getitem: " + str(idx))

        sequence_list = self.sequences[idx]
        img_list = []

        # get the label of the image using the label as "index" of the first image of the sequence. The label should
        # be consistent in all the sequence, this will be ensured later
        label = self.labels[self.images.index(sequence_list[0])]

        # flag used to print warning in the loop
        warning_flag = True

        if self.min_elements < min([len(v) for k, v in self.sequences.items()]):
            print("the specified min.elements for this dataloader smaller than the min-sequence-list of all "
                  "loaded sequences")

        if self.fixed_lenght == 1:
            # get the last min_elements
            sequence_list = sequence_list[self.min_elements:]

        if self.fixed_lenght == 2:
            # linearize space and take the elements with equal-spaces
            sequence_list = [sequence_list[i] for i in
                             np.round(np.linspace(0, len(sequence_list), self.min_elements, endpoint=False)).astype(
                                 int)]

        for path in sequence_list:

            # ensure the labels are consistent within the sequence
            if label != self.labels[self.images.index(path)]:
                assert 0, "Inconsistent label in sequence"

            # if (previous_label is not None) and (label != previous_label):
            #     print('Error in file: {}\n'.format(path))
            #     print('Sequence labels are not consistents')
            #     exit(-1)
            if self.all_in_ram:
                image = self.images_in_ram[path].copy()
            else:
                # image = Image.open(path)
                img_ = Image.open(path)
                image = img_.copy()
                img_.close()

            if not self.usePIL:
                # copy to avoid warnings from pytorch, or bad edits ...
                image = np.copy(np.asarray(image))

            if self.transform:
                image = self.transform(image)
            else:
                if warning_flag:
                    print('Class: ' + __class__.__name__ + " Warning: no transform passed. Sure?")
                    warning_flag = False

            img_list.append(image)

        sample = {'sequence': img_list, 'label': label, 'path_of_original_images': sequence_list}

        if len(img_list) == 1:
            print("LENGHT OF IMAGE LIST IS 1 !!!!! TAKE CARE!!!!")

        return sample

    @staticmethod
    def __get_sequences(self, image_path, filelist, last_seq, seq_dict):
        """

        Args:
            image_path: folder of the images
            filelist: list of files in the specific folder.
            last_seq: this function will be called more than one time. this variable is used as index.
            seq_dict: this list-of-lists will contain the intersection sequences

        Returns:

            the list of intersections, and for each of the intersections the associated frame filenames.

        """
        sq = last_seq
        sequence = []
        prev_framenumber = None
        for file in filelist:
            # take into account also the 'warpings' folder that contains appendix numbers like
            # framenumber.data-augmentation-counter.png like 0000000084.001.png
            # the following trick does the job, but it doesn't work for folders with multiple
            # data augmentation files, ie, filename.002+.png won't be handled correctly.
            # TODO: improve the creation of sequences from data-augmented folders.

            if '2013_' in file:
                # BRUTAL kitti360 patch: ... the frame numbering is different since they contain "folder" in the
                # filename itself... i'll try to catch this searching '2013_' in the string ...
                # 6/2013_05_28_drive_0009_sync_0000007162.001.png
                frame_number = int(os.path.splitext(os.path.split(file)[1])[0].split('.')[0].split('_')[-1])
            else:
                frame_number = int(os.path.splitext(os.path.split(file)[1])[0].split('.')[0])

            # check for sequence. if the current frame number is not the previous+1, then we have a new sequence.
            if not (prev_framenumber is None or (frame_number == (prev_framenumber + 1))):
                seq_dict[sq] = sequence.copy()
                sequence.clear()
                sq += 1

            sequence.append(os.path.join(image_path, file))
            prev_framenumber = frame_number

        # check if we have something that need to add
        if sequence:
            seq_dict[sq] = sequence.copy()
            sequence.clear()
            sq += 1


        # retrieve the min element of all sequences; will be used for LSTM fixed length eval
        min_elements = min([len(v) for k, v in seq_dict.items()])

        if self.verbose:
            print("SequencesDataloader, loaded folder: ", image_path)
            print("Found", len(seq_dict), " sequences; for each sequence, the associated frames are: ")
            print([len(v) for k, v in seq_dict.items()])
            print("Min elements along all sequences: ", str(min_elements))

        return seq_dict, sq, min_elements

    @staticmethod
    # TODO unused function here
    def __get_label(path):
        head, tail = os.path.split(path)
        head, _ = os.path.split(head)
        gt_path = os.path.join(head, 'frames_topology.txt')
        filename, _ = os.path.splitext(tail)
        gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        label = int(gtdata.loc[gtdata[0] == filename][2])

        return label


class SequencesDataloader(AbstractSequence, Dataset):
    """
    This dataloader is used to load sequences of the ALCALA dataset **ONLY**

    The Alcala dataset was recorded in 2019, and consists of three folders
    with data from 2 cameras, front/back. Was recorded when Daniele was here
    as visiting student; most of the dataset is "urban" in downtown Alcala.

    (base) ballardini@ballardini-T14:~/Desktop/ALCALA$ tree -d
    .
    ├── R1_video_0002_camera1_png
    ├── R2_video_0002_camera1_png
    │ └── image_02
    └── R2_video_0002_camera2_png

    there's one script "checkSequenceDataloader.py" we used to test this class, basically

    dataset = SequencesDataloader(root='/home/ballardini/Desktop/ALCALA/',
                              folders=['R2_video_0002_camera1_png'])

    """

    def __init__(self, root, folders, transform=None, isSequence=True):
        """

        Args:
            root: common path, for all folders
            folders: list of folders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(isSequence=isSequence)

        self.transform = transform
        sequences = {}
        last_seq = 0
        for folder in folders:
            image_path = os.path.join(root, os.path.join(folder, 'image_02'))
            filelist = glob.glob1(image_path, '*.png')
            filelist.sort()
            sequences, last_seq = self.__get_sequences(image_path, filelist, last_seq, sequences)

        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """

        Args:
            idx: index of the sequence. this is generated by the dataloader

        Returns:

            the sample containing the image list and the label of the sequence. if not all the frames have the same
            label, an "exit" will be called.

        """
        sequence_list = self.sequences[idx]
        img_list = []
        previous_label = None

        # flag used to print warning in the loop
        warning_flag = True

        for path in sequence_list:
            label = self.__get_label(path)

            # ensure the labels are consistent within the sequence
            if (previous_label is not None) and (label != previous_label):
                print('Error in file: {}\n'.format(path))
                print('Sequence labels are not consistents')
                exit(-1)
            image = Image.open(path)

            if self.transform:
                image = self.transform(image)
            else:
                if warning_flag:
                    print('Class: ' + __class__.__name__ + " Warning: no transform passed. Sure?")
                    warning_flag = False
            img_list.append(image)
            previous_label = label

        sample = {'sequence': img_list, 'label': previous_label}

        if self.transform:  # No se si esto funciona!
            sample['sequence'] = self.transform(sample['sequence'])

        return sample

    @staticmethod
    def __get_sequences(image_path, filelist, last_seq, seq_dict):
        """

        Args:
            image_path: folder of the images
            filelist: list of files in the specific folder.
            last_seq: this function will be called more than one time. this variable is used as index.
            seq_dict: this list-of-lists will contain the intersection sequences

        Returns:

            the list of intersections, and for each of the intersections the associated frame filenames.

        """
        sq = last_seq
        sequence = []
        prev_framenumber = None
        for file in filelist:
            frame_number = int(os.path.splitext(file)[0])

            # check for sequence. if the current frame number is not the previous+1, then we have a new sequence.
            if not (prev_framenumber is None or (frame_number == (prev_framenumber + 1))):
                seq_dict[sq] = sequence.copy()
                sequence.clear()
                sq += 1

            sequence.append(os.path.join(image_path, file))
            prev_framenumber = frame_number

        print("SequencesDataloader, loaded folder: ", image_path)
        print("Found", len(seq_dict), " sequences; for each sequence, the associated frames are: ")
        print([len(v) for k, v in seq_dict.items()])
        return seq_dict, sq

    @staticmethod
    def __get_label(path):
        head, tail = os.path.split(path)
        head, _ = os.path.split(head)
        gt_path = os.path.join(head, 'frames_topology.txt')
        filename, _ = os.path.splitext(tail)
        gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        label = int(gtdata.loc[gtdata[0] == filename][2])

        return label


#class txt_dataloader_styleGAN(txt_dataloader):
class txt_dataloader_styleGAN(lstm_txt_dataloader):
    """
    Adapts txt_dataloader to the structure of Pycharm datasets.ImageFolder
    """
    def __init__(self, path_filename_list=None, transform=None, usePIL=True, isSequence=False, decimateStep=1):

        version = 2

        if version == 1:
            # version 1 -- stylegan with all the images from the passed txt files; no decimate
            txt_dataloader.__init__(self, path_filename_list, transform, usePIL, isSequence)
            self.imgs = list(zip(self.images, self.labels))

        if version == 2:
            # version 2 -- stylegan with decimated 'per-sequence' list
            lstm_txt_dataloader.__init__(self, path_filename=path_filename_list, transform=transform, usePIL=True,
                                         isSequence=True, all_in_ram=False, fixed_lenght=0, verbose=True)

            # with this flag, the lstm_txt_dataloader GETITEM will change how it works
            self.GANflag = True

            # create a list of images in a way that:
            #   1. use the sequences instead of all the list of frames
            #   2. decimate each list of frames for each of the sequences.
            images_decimated = []
            labels_decimated = []

            for key in self.sequences:

                # if decimatestep > 1, then we want to decimate, BUT, alcala and kitti have different fps ... so i have
                # to do something like this, checking which sequence each of them belongs to...
                if decimateStep > 1:
                    if 'alcala' in self.sequences[key][0]:
                        images_decimated.append(self.sequences[key][::30])  # 30 FPS alcala sequences
                    elif 'KITTI' in self.sequences[key][0]:
                        images_decimated.append(self.sequences[key][::10])  # 10 FPS kitti sequences
                    else:
                        print('mmm... decimate does not work')
                        exit(-1)
                else:
                    images_decimated.append(self.sequences[key][::decimateStep])


            images_decimated = [item for sublist in images_decimated for item in sublist]

            # search the label, given each of the image filenames.. this is for compatibility
            for image in images_decimated:
                labels_decimated.append(self.labels[self.images.index(image)])

            print('*************************************')
            print('Debug info:')

            print('*************************************')
            print('Number of images: ' + str(len(images_decimated)))
            print('Number of labels: ' + str(len(labels_decimated)))
            print(dict(Counter(labels_decimated)))
            print('*************************************')

            # this list is used with 'generate.py' , added for compatibility only; other behavior = directly use the
            # the pytorch dataloader with this dataset
            self.imgs = list(zip(images_decimated, labels_decimated))

    def __len__(self):
        return txt_dataloader.__len__(self)

    def __getitem__(self, idx):
        sample_ = txt_dataloader.__getitem__(self, idx)
        sample = (sample_['data'], sample_['label'])
        return sample
