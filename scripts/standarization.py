import os
import numpy as np

from dataloaders.sequencedataloader import fromGeneratedDataset
from torch.utils.data import DataLoader

data_path = '/home/malvaro/Documentos/DualBiSeNet/data_raw_bev/'

# All sequence folders
folders = np.array([os.path.join(data_path, folder) for folder in os.listdir(data_path) if
                    os.path.isdir(os.path.join(data_path, folder))])

dataset = fromGeneratedDataset(folders, distance=20.0, loadlist=True)

loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False,
    drop_last=True
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    image = data['data']
    image = image.permute(0, 3, 1, 2)
    image = image.float()
    batch_samples = image.size(0)
    image = image.view(batch_samples, image.size(1), -1)
    mean += image.mean(2).sum(0)
    std += image.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('Mean: {}'.format(mean))
print('Std: {}'.format(std))
