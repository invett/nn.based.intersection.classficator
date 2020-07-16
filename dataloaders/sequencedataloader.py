# Dataloader for DualBisenet under prepared Kitti dataset
import os
from cv2 import imread
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

    def __len__(self):

        return len(self.file_list)

    def __getitem__(self, idx):

        # Select file subset
        imagepath = self.file_list[idx]

        image = imread(imagepath, cv2.IMREAD_UNCHANGED)

        # Obtaining ground truth
        head, tail = os.path.split(imagepath)
        filename, _ = os.path.splitext(tail)
        gt_path = os.path.join(head, 'frames_topology.txt')
        data = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
        gTruth = int(data.loc[0] == filename)

        sample = {'data': image, 'label': gTruth}

        if self.transform:
            sample = self.transform(sample)

        return sample
