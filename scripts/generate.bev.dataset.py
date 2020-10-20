import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import sys
import time

import torchvision.transforms as transforms
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev, Mirror, GenerateNewDataset, \
    WriteDebugInfoOnNewDataset, GenerateWarping
from dataloaders.sequencedataloader import fromAANETandDualBisenet, fromAANETandDualBisenet360
import matplotlib.pyplot as plt
from miscellaneous.utils import send_telegram_message, send_telegram_picture

# SCRIPT THAT USES THE DATALOADER TO GENERATE THE AUGMENTED-DATASET OF BEVs
# Use this line to delete **ALL FILES** in current folder and subfolders
# as /home/malvaro/Documentos/DualBiSeNet/data_raw_bev
# find . -name "*" ! -name "*.txt"  -type f             <<< LIST THE FILES
# find . -name "*" ! -name "*.txt"  -type f -delete     <<< DELETE THOSE FILES


def main(args):

    folders = np.array([os.path.join(args.rootfolder, folder) for folder in sorted(os.listdir(args.rootfolder)) if
                        os.path.isdir(os.path.join(args.rootfolder, folder))])

    #folders = [folders[0]]

    #execute = 'warping'
    #execute = 'standard'
    execute = 'kitti360'

    if execute == 'kitti360':

        dataset = fromAANETandDualBisenet360(folders, transform=transforms.Compose([#Normalize(),
                                                                                 GenerateBev(returnPoints=False,
                                                                                             max_front_distance=args.max_front_distance,
                                                                                             max_height=args.max_height,
                                                                                             excludeMask=args.excludeMask,
                                                                                             decimate=1.0,
                                                                                             random_Rx_degrees=args.random_Rx_degrees,
                                                                                             random_Ry_degrees=args.random_Ry_degrees,
                                                                                             random_Rz_degrees=args.random_Rz_degrees,
                                                                                             random_Tx_meters=args.random_Tx_meters,
                                                                                             random_Ty_meters=args.random_Ty_meters,
                                                                                             random_Tz_meters=args.random_Tz_meters,
                                                                                             qmatrix='kitti360'
                                                                                             ),
                                                                                 Mirror(),
                                                                                 Rescale((224, 224)),
                                                                                 WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]),
                                             distance=args.distance_from_intersection)

    if execute == 'warping':
        # WARNING! MIRROR IS/WAS DISABLED! not sure whether this respects our intentions...
        dataset = fromAANETandDualBisenet(folders, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=1.0,
                                                                                                 random_Ry_degrees=1.0,
                                                                                                 random_Rz_degrees=1.0,
                                                                                                 random_Tx_meters=5.0,
                                                                                                 random_Ty_meters=1.0,
                                                                                                 random_Tz_meters=0.1),
                                                                                 #Mirror(),
                                                                                 Rescale((224, 224)),
                                                                                 WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]),
                                          distance=args.distance_from_intersection)

    if execute == 'standard':
        dataset = fromAANETandDualBisenet(folders, transform=transforms.Compose([#Normalize(),
                                                                                 GenerateBev(returnPoints=False,
                                                                                             max_front_distance=args.max_front_distance,
                                                                                             max_height=args.max_height,
                                                                                             excludeMask=args.excludeMask,
                                                                                             decimate=1.0,
                                                                                             random_Rx_degrees=args.random_Rx_degrees,
                                                                                             random_Ry_degrees=args.random_Ry_degrees,
                                                                                             random_Rz_degrees=args.random_Rz_degrees,
                                                                                             random_Tx_meters=args.random_Tx_meters,
                                                                                             random_Ty_meters=args.random_Ty_meters,
                                                                                             random_Tz_meters=args.random_Tz_meters
                                                                                             ),
                                                                                 Mirror(),
                                                                                 Rescale((224, 224)),
                                                                                 WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]),
                                          distance=args.distance_from_intersection)

    # num_workers starts from 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    for index in range(args.augmentation+1):
        print("Generating run {} ... ".format(index))
        if args.telegram:
            send_telegram_message("Generating run {} ... ".format(index))

        # RESET SEED to have np.random working correctly https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        for sample in dataloader:
            data = sample['data']
            label = sample['label']
            if args.telegram:
                a = plt.figure()
                plt.imshow(sample['data'].numpy().squeeze() / 255.0)
                send_telegram_picture(a, str(sample['bev_path_filename']))
                plt.close('all')

            if 'bev_path_filename' in sample:
                print(sample['bev_path_filename'])

            #break

        print("Run {} generated".format(index))
        if args.telegram:
            send_telegram_message("Run {} generated successfully".format(index))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--rootfolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw", type=str, help='Root folder for all datasets')
    # parser.add_argument('--savefolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw_bev", type=str, help='Where to save the new data')
    parser.add_argument('--rootfolder', default="/media/augusto/500GBHECTOR/augusto/kitti360-augusto", type=str, help='Root folder for all datasets')
    parser.add_argument('--savefolder', default="/media/augusto/500GBHECTOR/augusto/kitti360-augusto-augmented", type=str, help='Where to save the new data')
    parser.add_argument('--augmentation', type=int, default=50, help='How many files generate for each of the BEVs')
    parser.add_argument('--workers', type=int, default=0, help='How many workers for the dataloader')
    parser.add_argument('--telegram', action='store_true', help='Send info through Telegram')
    parser.add_argument('--debug', action='store_true', help='Print filenames as walking the filesystem')
    parser.add_argument('--max_front_distance', type=float, default=50.0, help='Distance from the cross')
    parser.add_argument('--max_height', type=float, default=10.0, help='desired up-to-height from the ground')
    parser.add_argument('--distance_from_intersection', type=float, default=20.0, help='Distance from the cross')
    parser.add_argument('--excludeMask', action='store_true', help='If true, don\'t mask the images with 3D-DEEP')

    parser.add_argument('--random_Rx_degrees', type=float, default=0.0, help='random_Rx_degrees')
    parser.add_argument('--random_Ry_degrees', type=float, default=0.0, help='random_Ry_degrees')
    parser.add_argument('--random_Rz_degrees', type=float, default=0.0, help='random_Rz_degrees')
    parser.add_argument('--random_Tx_meters',  type=float, default=0.0, help='random_Tx_meters')
    parser.add_argument('--random_Ty_meters',  type=float, default=0.0, help='random_Ty_meters')
    parser.add_argument('--random_Tz_meters',  type=float, default=0.0, help='random_Tz_meters')

    # random_Rx_degrees = 2.0,
    # random_Ry_degrees = 15.0,
    # random_Rz_degrees = 2.0,
    # random_Tx_meters = 2.0,
    # random_Ty_meters = 2.0,
    # random_Tz_meters = 2.0,

    args = parser.parse_args()

    if args.telegram:
        send_telegram_message("Executing generate.bev.dataset.py")

    try:
        tic = time.time()
        main(args)
        toc = time.time()
        if args.telegram:
            send_telegram_message("Generation of dataset ended correctly after " +
                                  str(time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

    except (KeyboardInterrupt, SystemExit):
        print("Shutdown requested")
        if args.telegram:
            send_telegram_message("Shutdown requested")
        raise
    except:
        e = sys.exc_info()
        print(e)
        if args.telegram:
            send_telegram_message("Error generating the dataset" + str(e))
