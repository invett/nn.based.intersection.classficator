

###
###
###
###     this little script was used to check/develope the sequence dataloader. nothing special here.
###
###
###

import os
import numpy as np

from dataloaders.sequencedataloader import SequencesDataloader
from torch.utils.data import DataLoader

data_path = '/home/ballardini/Desktop/ALCALA/R1_video_0002_camera1_png/'
data_path = ['/media/ballardini/7D3AD71E1EACC626/ALVARO/Secuencias/2011_10_03_drive_0027_sync/']
data_path = ['/home/ballardini/Desktop/alcala-26.01.2021/']

# All sequence folders
# folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
#                    os.path.isdir(os.path.join(data_path, folder))])

#dataset = SequencesDataloader(root='/media/ballardini/7D3AD71E1EACC626/ALVARO/Secuencias/',
#                              folders=['2011_10_03_drive_0027_sync'])

# dataset = SequencesDataloader(root='/home/ballardini/Desktop/ALCALA/',
#                               folders=['R2_video_0002_camera1_png'])

dataset = SequencesDataloader(root='/home/ballardini/Desktop/alcala-26.01.2021/',
                              folders=['161604AA', '161657AA', '161957AA', '162257AA', '162557AA', '162857AA',
                                       '163157AA', '163457AA', '163757AA', '164057AA', '164357AA', '164657AA',
                                       '164957AA', '165257AA', '165557AA', '165857AA', '170157AA', '170457AA',
                                       '170757AA', '171057AA', '171357AA', '171657AA', '171957AA', '172257AA',
                                       '172557AA', '172857AA', '173158AA', '173457AA', '173757AA', '174057AA',
                                       '174357AA', '174657AA', '174957AA', '175258AA', '175557AA', '175857AA',
                                       '180158AA', '180458AA', '180757AA', '181058AA', '181358AA', '181658AA',
                                       '181958AA', '182258AA', '182558AA', '182858AA', '183158AA', '183458AA',
                                       '183758AA', '184058AA', '184358AA', '184658AA'],
                              suffixPath='')

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False
)

for idx, data in enumerate(loader):
        print(idx)

print("End")
