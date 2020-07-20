# Dataloader for DualBisenet under prepared Kitti dataset
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from numpy import load
import cv2


class TestDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
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

        if self.transform:
            bev_with_new_label = self.transform(sample)

        return bev_with_new_label
