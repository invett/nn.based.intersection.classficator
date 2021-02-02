

###
###
###
###     this little script was used to check/develope the sequence dataloader. nothing special here.
###
###
###

import os
import numpy as np

from dataloaders.sequencedataloader import SequencesDataloader, alcala26012021
from torch.utils.data import DataLoader

data_path = '/home/ballardini/Desktop/ALCALA/R1_video_0002_camera1_png/'
data_path = ['/media/ballardini/7D3AD71E1EACC626/ALVARO/Secuencias/2011_10_03_drive_0027_sync/']
data_path = '/home/ballardini/Desktop/alcala-26.01.2021/'

# All sequence folders
# folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
#                    os.path.isdir(os.path.join(data_path, folder))])

#dataset = SequencesDataloader(root='/media/ballardini/7D3AD71E1EACC626/ALVARO/Secuencias/',
#                              folders=['2011_10_03_drive_0027_sync'])

# dataset = SequencesDataloader(root='/home/ballardini/Desktop/ALCALA/',
#                               folders=['R2_video_0002_camera1_png'])

dataset = alcala26012021(path_filename='/home/ballardini/Desktop/alcala-26.01.2021/train_list.txt')

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False
)

for idx, data in enumerate(loader):
        print(idx)

print("End")
