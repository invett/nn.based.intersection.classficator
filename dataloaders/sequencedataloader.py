# Dataloader for DualBisenet under prepared Kitti dataset
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
from numpy import load
import cv2
import json
from miscellaneous.utils import write_ply
import random
import time
import numpy as np

from scripts.OSM_generator import Crossing, test_crossing_pose
from miscellaneous.utils import send_telegram_picture
import matplotlib.pyplot as plt

class BaseLine(Dataset):
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
                if os.path.isfile(os.path.join(folder_image_02, image_02_file)) and '.png' in image_02_file:
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
        FOR THESE REASON THEN WE MADE AN "AUGMENTED" OFFLINE DATASET WITH A NEW DATALOADER, <fromGeneratedDataset>

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
            base_file_star = str(file.split(".")[0]) + "*"
            current_filelist = glob.glob1(folder, base_file_star)
            last_number = len([x for x in current_filelist if "json" not in x])
            final_filename = str(file.split(".")[0]) + '.' + str(last_number + 1).zfill(3) + ".png"
            bev_path_filename = os.path.join(folder, final_filename)

            # path must already exist!
            cv2.imwrite(bev_path_filename, bev_with_new_label['data'])
            bev_with_new_label['bev_path_filename'] = bev_path_filename

            # for debuggin' purposes
            if "save_out_points" in bev_with_new_label:
                cv2.imwrite(bev_path_filename + "original.png", image_02_image)
                write_ply(bev_path_filename + ".ply", bev_with_new_label['save_out_points'],
                          bev_with_new_label['save_out_colors'])

            if "debug" in bev_with_new_label:
                debug_values = bev_with_new_label.copy()
                debug_values.pop('data')
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
                 rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True, canonical=True, addGeneratedOSM=True,
                 decimateStep=1, savelist=False, loadlist=False):
        # TODO addGeneratedOSM should not be the default behavior; set FALSE here anc call with TRUE as needed

        """
        Args:
            root_dir (string): Directory with all the images.

            # THE FOLLOWING PARAMETERS ARE COPIED FROM <teacher_tripletloss_generated>
            rnd_width: parameter for uniform noise add (width);          uniform(-rnd_width, rnd_width)
            rnd_angle: parameter for uniform noise add (rotation [rad]); uniform(-rnd_angle, rnd_angle)
            rnd_spatial: parameter for uniform spatial cross position (center of the crossing area)
            canonical: set false to avoid generating the "canonical" crossings
            noise: whether to add or not noise in the image (pixel level)
            decimateStep: use this value to decimate the 100-elements of the generated-dataset; set as "::STEP"

            addGeneratedOSM: whether to add the generated OSM intersection (to train as student)

        """

        if not isinstance(decimateStep, int) and decimateStep > 0:
            print("decimateStep must be an integer > 0")
            exit(-1)
            
        if savelist and loadlist:
            print("load and save at the same time")
            exit(-1)

        self.transform = transform

        self.bev_images = []
        self.bev_labels = []

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.canonical = canonical
        self.addGeneratedOSM = addGeneratedOSM

        tic = time.time()

        fullfilename = os.path.join(os.path.commonpath(folders.tolist()),
                                    "fromGeneratedDataset" + str(distance) + ".npz")

        if loadlist and os.path.isfile(fullfilename):

            print("Loading existing file list from " + fullfilename)

            self.bev_images = np.load(fullfilename)['bev_images'].tolist()
            self.bev_labels = np.load(fullfilename)['bev_labels'].tolist()

        else:

            if os.path.isfile(fullfilename):
                print("A saved version of the datasets exists. Consider using --loadlist")

            for folder in folders:
                current_filelist = glob.glob1(str(folder), "*.png")
                for file in current_filelist:
                    bev_filename = os.path.join(folder, file)
                    json_filename = str(bev_filename.replace('png', 'png.json'))

                    #assert os.path.exists(bev_filename), "no bev file"
                    #assert os.path.exists(json_filename), "no json file"

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

        toc = time.time()
        print("[fromGeneratedDataset] - " + str(len(self.bev_labels)) + " elements loaded in " + str(
            time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

    def __len__(self):

        return len(self.bev_labels)

    def __getitem__(self, idx):

        bev_image = cv2.imread(self.bev_images[idx], cv2.IMREAD_UNCHANGED)
        bev_label = self.bev_labels[idx]

        if self.addGeneratedOSM:
            # Sample an intersection given a label; this is used in the STUDENT training
            generated_osm = test_crossing_pose(crossing_type=bev_label, save=False, rnd_width=self.rnd_width,
                                               rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise)
            sample = {'data': bev_image,
                      'label': bev_label,
                      'image_path': self.bev_images[idx],
                      'generated_osm': generated_osm[0]}  # TODO this [0] might be a bug
        else:
            sample = {'data': bev_image,
                      'label': bev_label,
                      'image_path': self.bev_images[idx]
                      }

        # debug code to send the sample over telegram
        # a = plt.figure()
        # plt.imshow(np.asarray(sample['data'], dtype=np.uint8))
        # send_telegram_picture(a, "data")
        # plt.imshow(np.asarray(sample['generated_osm'], dtype=np.uint8))
        # send_telegram_picture(a, "generated_osm")

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __filterdistance(self, distance):
        images = []
        labels = []
        datapath_last = ""
        for file, label in zip(self.bev_images, self.bev_labels):
            head, filename = os.path.split(file)
            head = head.replace('data_raw_bev', 'data_raw')
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

    def __init__(self, folders, distance, include_insidecrossing=False, transform=None, noise=True, canonical=True):
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
                                                 sampling=False)[0]
        else:
            canonical_image = 0

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
                 transform=None, random_rate=1.0):
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

        """

        self.rnd_width = rnd_width
        self.rnd_angle = rnd_angle
        self.rnd_spatial = rnd_spatial
        self.noise = noise
        self.canonical = canonical
        self.random_rate = random_rate

        self.transform = transform

        # Generate a list of crossings; will be then used during the sampling process; is just a list of int s
        osm_types = []

        for crossing_type in range(0, 7):
            for _ in range(0, elements):
                # sample = test_crossing_pose(crossing_type=crossing_type, noise=True, rnd_width=1.0, save=False)
                osm_types.append([crossing_type])

        self.samples = osm_types

    def __len__(self):
        return len(self.samples)

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

        # identify the typology for anchor item
        anchor_type = self.samples[idx][0]

        # create positive and negative list based on the anchor item typology
        positive_list = [element for element in self.samples if element[0] == anchor_type]
        negative_list = [element for element in self.samples if element[0] != anchor_type]

        # remove the anchor item from the positive list
        positive_list.remove(self.samples[idx])
        positive_item = random.choice(positive_list)
        negative_item = random.choice(negative_list)

        anchor_image = test_crossing_pose(crossing_type=anchor_type, save=False, rnd_width=self.rnd_width,
                                          rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise)
        positive_image = test_crossing_pose(crossing_type=positive_item[0], save=False, rnd_width=self.rnd_width,
                                            rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise)
        negative_image = test_crossing_pose(crossing_type=negative_item[0], save=False, rnd_width=self.rnd_width,
                                            rnd_angle=self.rnd_angle, rnd_spatial=self.rnd_spatial, noise=self.noise)
        if self.canonical:  # set canonical to False to speedup this dataloader
            canonical_image = test_crossing_pose(crossing_type=anchor_type, save=False, noise=self.noise,
                                                 sampling=False)
        else:
            canonical_image = [0]

        # anchor_image = cv2.imread(self.samples[idx][0], cv2.IMREAD_UNCHANGED)

        sample = {'anchor': anchor_image[0],  # [0] is the image
                  'positive': positive_image[0],
                  'negative': negative_image[0],
                  'canonical': canonical_image[0],
                  'label_anchor': anchor_type,
                  'label_positive': positive_item[0],  # [0] is the type
                  'label_negative': negative_item[0],  # [0] is the type
                  'ground_truth_image': anchor_image[0],  # for debugging purposes | in this dataloader is = the anchor
                  'anchor_xx': anchor_image[1],  # [1] is the xx coordinate
                  'anchor_yy': anchor_image[2],  # [2] is the yy coordinate
                  'positive_xx': positive_image[1],
                  'positive_yy': positive_image[2],
                  'negative_xx': negative_image[1],
                  'negative_yy': negative_image[2],

                  # the following are not used; are here to mantain the compatibility with "teacher_tripletloss" 
                  'filename_anchor': 0, 'filename_positive': 0, 'filename_negative': 0, 'anchor_oxts_lat': 0,
                  'anchor_oxts_lon': 0, 'positive_oxts_lat': 0, 'positive_oxts_lon': 0, 'negative_oxts_lat': 0,
                  'negative_oxts_lon': 0}

        if self.transform:
            sample['anchor'] = self.transform(sample['anchor'])
            sample['positive'] = self.transform(sample['positive'])
            sample['negative'] = self.transform(sample['negative'])

        return sample


class triplet_OBB(teacher_tripletloss_generated, fromGeneratedDataset, Dataset):

    def __init__(self, folders, distance, elements=1000, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
                 canonical=True, transform=None, random_rate=1.0, loadlist=True, savelist=False, decimateStep=1):

        teacher_tripletloss_generated.__init__(self, elements=elements, rnd_width=rnd_width, rnd_angle=rnd_angle,
                                               rnd_spatial=rnd_spatial, noise=noise, canonical=canonical,
                                               transform=transform, random_rate=random_rate)

        fromGeneratedDataset.__init__(self, folders, distance, transform=transform, rnd_width=rnd_width,
                                      rnd_angle=rnd_angle, rnd_spatial=rnd_spatial, noise=noise, canonical=canonical,
                                      addGeneratedOSM=True, decimateStep=decimateStep,
                                      savelist=savelist, loadlist=loadlist)

    def __len__(self):
        # In this multi inherit class we have both [teacher_tripletloss_generated] and [fromGeneratedDataset] items.
        # in OBB , OSM + BEV + BEV we want that our list of elements is like the teacher_tripletloss_generated one.
        return len(self.samples)

    def __getitem__(self, idx):

        # OBB ... so to get the OSM we can simply get the anchor from the teacher_tripletloss_generated triplet
        sample_ttg = teacher_tripletloss_generated.__getitem__(self, idx)

        OSM = sample_ttg['anchor']

        # for the POSITIVE and NEGATIVE, we can extract the indices from the self.bev_labels using the OSM anchor label
        positive_list = [idx for idx, element in enumerate(self.bev_labels) if element == sample_ttg['label_anchor']]
        negative_list = [idx for idx, element in enumerate(self.bev_labels) if element != sample_ttg['label_anchor']]

        positive_item = random.choice(positive_list)
        negative_item = random.choice(negative_list)

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

        # DEBUG -- send in telegram. Little HACK for the ANCHOR, get a sub-part of image ...
        emptyspace = 255 * torch.ones([224, 30, 3], dtype=torch.uint8)
        a = plt.figure()
        plt.imshow(torch.cat((torch.tensor(sample['OSM_anchor'])[38:262, 38:262, :], emptyspace,
                              torch.tensor(sample['BEV_positive']), emptyspace, torch.tensor(sample['BEV_negative'])),
                             1))
        send_telegram_picture(a, "OSM | BEV(positive) | BEV(negative)" +
                              "\n __image of OSM_anchor is CROPPED__\n" +
                              "\nlabel_anchor: " + str(sample['label_anchor']) +
                              "\nlabel_positive: " + str(sample['label_positive']) +
                              "\nlabel_negative: " + str(sample['label_negative']) +
                              "\nfilename positive: " + str(sample['filename_positive']) +
                              "\nfilename negative: " + str(sample['filename_negative'])
                              )
        
        return sample
