# Dataloader for DualBisenet under prepared Kitti dataset
import os
import cv2
from torch.utils.data import Dataset
import pandas as pd


class SequenceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        folders = [os.path.join(self.root_dir, folder) for folder in os.listdir(self.root_dir) if
                   os.path.isdir(os.path.join(self.root_dir, folder))]
        filelist = []
        for folder in folders:
            folder = os.path.join(folder, 'bev')
            for file in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, file)) and '.png' in file:
                    filelist.append(os.path.join(folder, file))

        self.file_list = filelist
        assert len(self.file_list) > 0, 'Training files missing'

    def __len__(self):

        return len(self.file_list)

    def __getitem__(self, idx):

        # Select file subset
        imagepath = self.file_list[idx]

        image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)

        # Obtaining ground truth
        head, tail = os.path.split(imagepath)
        head, _ = os.path.split(head)
        filename, _ = os.path.splitext(tail)
        gt_path = os.path.join(head, 'frames_topology.txt')
        gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        gTruth = int(gtdata.loc[gtdata[0] == filename][2])

        sample = {'data': image, 'label': gTruth}

        if self.transform:
            sample = self.transform(sample)

        return sample
