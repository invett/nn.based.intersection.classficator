import os
import numpy as np

from dataloaders.sequencedataloader import triplet_OBB
from torch.utils.data import DataLoader

data_path = '/home/malvaro/Documentos/DualBiSeNet/data_raw_bev/'

# All sequence folders
folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                    os.path.isdir(os.path.join(data_path, folder))])

dataset = triplet_OBB(folders, distance=20.0, loadlist=True)

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=True
)

for idx, data in enumerate(loader):
    print(idx)

print("End")
