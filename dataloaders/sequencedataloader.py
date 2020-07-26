# Dataloader for DualBisenet under prepared Kitti dataset
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from numpy import load
import cv2
import json
from miscellaneous.utils import write_ply


class BaseLine(Dataset):
    def __init__(self, folders, transform=None):
        """
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

        sample = {'data': image, 'label': gTruth}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample


class TestDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
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


class fromAANETandDualBisenet(Dataset):

    def __init__(self, folders, transform=None):
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

                if os.path.isfile(os.path.join(folder_aanet, file)) and \
                        os.path.isfile(os.path.join(folder_alvaromask, alvarofile)) and \
                        os.path.isfile(os.path.join(folder_image_02, image_02_file)):
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
