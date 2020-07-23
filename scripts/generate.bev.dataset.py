import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import requests
import json

import torchvision.transforms as transforms
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev, Mirror, GenerateNewDataset
from dataloaders.sequencedataloader import fromAANETandDualBisenet

telegram_token = "1178257144:AAH5DEYxJjPb0Qm_afbGTuJZ0-oqfIMFlmY"  # replace TOKEN with your bot's token
telegram_channel = '-1001352516993'


def send_telegram_message(message):
    """

    Args:
        message: text

    Returns: True if ok

    """
    URI = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + telegram_channel + '&parse_mode=Markdown&text=' + message
    response = requests.get(URI)
    return json.loads(response.content)['ok']


def main(args):

    folders = np.array([os.path.join(args.rootfolder, folder) for folder in os.listdir(args.rootfolder) if
                        os.path.isdir(os.path.join(args.rootfolder, folder))])

    dataset = fromAANETandDualBisenet(folders, transform=transforms.Compose([Normalize(),
                                                                             GenerateBev(),
                                                                             Mirror(),
                                                                             Rescale((224, 224)),
                                                                             GenerateNewDataset(args.savefolder)]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for sample in dataloader:
        data = sample['data']
        label = sample['label']

        print("gigi")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--rootfolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw", type=str, help='Root folder for all datasets')
    parser.add_argument('--savefolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw_bev", type=str, help='Where to save the new data')
    parser.add_argument('--augmentation', type=int, default=50, help='How many files generate for each of the BEVs')
    parser.add_argument('--telegram', type=bool, default=False, help='Send info through Telegram')
    parser.add_argument('--debug', default=False, type=bool, help='Print namefiles as walking the filesystem')

    args = parser.parse_args()

    if args.telegram:
        send_telegram_message("Executing generate.bev.dataset.py")

    main(args)
