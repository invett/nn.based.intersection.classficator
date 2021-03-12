import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import sys
import time

import torchvision.transforms as transforms
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev, Mirror, GenerateNewDataset, \
    WriteDebugInfoOnNewDataset, GenerateWarping, addEntry
from dataloaders.sequencedataloader import fromAANETandDualBisenet, fromAANETandDualBisenet360, txt_dataloader
import matplotlib.pyplot as plt
from miscellaneous.utils import send_telegram_message, send_telegram_picture

# SCRIPT THAT USES THE DATALOADER TO GENERATE THE AUGMENTED-DATASET OF BEVs
# Use this line to delete **ALL FILES** in current folder and subfolders
# as /home/malvaro/Documentos/DualBiSeNet/data_raw_bev
# find . -name "*" ! -name "*.txt"  -type f             <<< LIST THE FILES
# find . -name "*" ! -name "*.txt"  -type f -delete     <<< DELETE THOSE FILES


def main(args):

    # folders = [folders[1]]
    execute = args.execute
    if not execute:
        # execute = 'KITTI-ROAD-WARPING'
        execute = 'KITTI-ROAD-3D'
        # execute = 'kitti360'
        # execute = 'kitti360-warping'

        # execute = 'alcala26012021'   #CHANGE ALSO FILENAME OR USE ROOTFOLDER parser.add_argument('--rootfolder',
        # execute = 'alcala.12.02.2021.000'   #CHANGE ALSO FILENAME OR USE ROOTFOLDER parser.add_argument('--rootfolder',
        # execute = 'alcala.12.02.2021.001'  #CHANGE ALSO FILENAME OR USE ROOTFOLDER parser.add_argument('--rootfolder',

        # execute = 'warping'
        # execute = 'standard'

    # alcala26122012 does not walk os paths! it directly uses a .txt file!
    if execute != 'alcala26012021' and execute != 'alcala.12.02.2021.000' and execute != 'alcala.12.02.2021.001' and \
            execute != 'KITTI-ROAD-WARPING' and execute != 'kitti360-warping' and execute != 'KITTI-ROAD-3D':
        folders = np.array([os.path.join(args.rootfolder, folder) for folder in sorted(os.listdir(args.rootfolder)) if
                            os.path.isdir(os.path.join(args.rootfolder, folder))])

    if execute == 'kitti360-warping':
        '''
        in the warping, moving the camera doesn't make sense i think ... better to make standard image augmentation
        '''

        # PREVIOUS ATTEMPTS ..
        # dataset = fromAANETandDualBisenet360(folders, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.29,
        #                                                                                             random_Ry_degrees=0.0,
        #                                                                                             random_Rz_degrees=0.0,
        #                                                                                             random_Tx_meters=0.0,
        #                                                                                             random_Ty_meters=0.0,
        #                                                                                             random_Tz_meters=0.0,
        #                                                                                             warpdataset='kitti360'),
        #                                                                             Mirror(),
        #                                                                             Rescale((224, 224)),
        #                                                                             #WriteDebugInfoOnNewDataset(),
        #                                                                             GenerateNewDataset(args.savefolder)]
        #                                                                            ),
        #                                      distance=args.distance_from_intersection)

        # NOW DO THE SAME BUT WITH THE TXT FILES
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.2,
                                                                                                     random_Ry_degrees=0.5,
                                                                                                     random_Rz_degrees=0.5,
                                                                                                     random_Tx_meters=2.0,
                                                                                                     random_Ty_meters=1.0,
                                                                                                     random_Tz_meters=0.1,
                                                                                                     warpdataset='kitti360',
                                                                                                     ignoreAllGivenRandomValues=True),
                                                                                     Rescale((224, 224)),
                                                                                     #WriteDebugInfoOnNewDataset(),
                                                                                     GenerateNewDataset(args.savefolder)]), usePIL=False)

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
                                                                                             qmatrix='kitti360',
                                                                                             base_Tz=22.5,
                                                                                             base_Ty=22.0
                                                                                             ),
                                                                                 #Mirror(),
                                                                                 Rescale((224, 224)),
                                                                                 WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]),
                                             distance=args.distance_from_intersection)

    if execute == 'warping':
        # WARNING! MIRROR IS/WAS DISABLED! not sure whether this respects our intentions...
        dataset = fromAANETandDualBisenet(folders, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=1.0,
                                                                                                 random_Ry_degrees=1.0,
                                                                                                 random_Rz_degrees=1.0,
                                                                                                 random_Tx_meters=2.5,
                                                                                                 random_Ty_meters=1.0,
                                                                                                 random_Tz_meters=0.1,
                                                                                                 warpdataset='kitti'),
                                                                                 #Mirror(),
                                                                                 Rescale((224, 224)),
                                                                                 WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]),
                                          distance=args.distance_from_intersection)

    if execute == 'alcala26012021':
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.2,
                                                                                                 random_Ry_degrees=0.0, #0.2,
                                                                                                 random_Rz_degrees=0.0, #0.2,
                                                                                                 random_Tx_meters =0.0, #2.0,
                                                                                                 random_Ty_meters =0.0, #1.0,
                                                                                                 random_Tz_meters =0.0, #0.1,
                                                                                                 warpdataset='alcala26012021',
                                                                                                 ignoreAllGivenRandomValues=True),
                                                                                 Rescale((224, 224)),
                                                                                 #WriteDebugInfoOnNewDataset(),
                                                                                 GenerateNewDataset(args.savefolder)]), usePIL=False)

    if execute == 'alcala.12.02.2021.000':
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.2,
                                                                                                     random_Ry_degrees=0.5,
                                                                                                     random_Rz_degrees=0.5,
                                                                                                     random_Tx_meters=2.0,
                                                                                                     random_Ty_meters=1.0,
                                                                                                     random_Tz_meters=0.1,
                                                                                                     warpdataset='alcala-12.02.2021.000',
                                                                                                     ignoreAllGivenRandomValues=True),
                                                                                     Rescale((224, 224)),
                                                                                     #WriteDebugInfoOnNewDataset(),
                                                                                     GenerateNewDataset(args.savefolder)]), usePIL=False)

    if execute == 'alcala.12.02.2021.001':
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.2,
                                                                                                     random_Ry_degrees=0.5,
                                                                                                     random_Rz_degrees=0.5,
                                                                                                     random_Tx_meters=2.0,
                                                                                                     random_Ty_meters=1.0,
                                                                                                     random_Tz_meters=0.1,
                                                                                                     warpdataset='alcala-12.02.2021.001',
                                                                                                     ignoreAllGivenRandomValues=True),
                                                                                     Rescale((224, 224)),
                                                                                     #WriteDebugInfoOnNewDataset(),
                                                                                     GenerateNewDataset(args.savefolder)]), usePIL=False)

    if execute == 'KITTI-ROAD-WARPING':
        # this dataloader, uses the .txt files. Specify inside warpdataset the warping you need.
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([GenerateWarping(random_Rx_degrees=0.2,
                                                                                                     random_Ry_degrees=0.5,
                                                                                                     random_Rz_degrees=0.5,
                                                                                                     random_Tx_meters=2.0,
                                                                                                     random_Ty_meters=1.0,
                                                                                                     random_Tz_meters=0.1,
                                                                                                     warpdataset='KITTI-ROAD-WARPING',
                                                                                                     ignoreAllGivenRandomValues=True),
                                                                                     Rescale((224, 224)),
                                                                                     #WriteDebugInfoOnNewDataset(),
                                                                                     GenerateNewDataset(args.savefolder)]), usePIL=False)

    if execute == 'KITTI-ROAD-3D':
        # this dataloader, uses the .txt files. Specify inside warpdataset the warping you need.
        dataset = txt_dataloader(path_filename=args.rootfolder, transform=transforms.Compose([addEntry('aanet', 'no.value.needed.here'),
                                                                                             GenerateBev(returnPoints=False,
                                                                                             max_front_distance=args.max_front_distance,
                                                                                             max_height=args.max_height,
                                                                                             excludeMask=args.excludeMask,
                                                                                             decimate=1.0,
                                                                                             random_Rx_degrees=0.0,
                                                                                             random_Ry_degrees=0.0,
                                                                                             random_Rz_degrees=0.0,
                                                                                             random_Tx_meters =0.0,
                                                                                             random_Ty_meters =0.0,
                                                                                             random_Tz_meters =0.0,
                                                                                             txtdataloader=True,
                                                                                             qmatrix='kitti'
                                                                                             ),
                                                                                     Rescale((224, 224)),
                                                                                     GenerateNewDataset(args.savefolder)]), usePIL=False)

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
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=args.workers)

    # USE THIS TO SELECT ONE SINGLE IMAGE AND VERIFY THE WARPING
    # to see the path of the original image, enable # print(imagepath) in line def __getitem__(self, idx):
    # on file sequencedataloader.py
    # for i in range(12):
    #     # sample = dataloader.dataset.__getitem__(0)
    #     sample = dataloader.dataset.__getitem__(2013+i)
    #     data = sample['data']
    #     label = sample['label']
    #     a = plt.figure()
    #     plt.imshow(sample['data'] / 255.0)
    #     send_telegram_picture(a, '')
    #     plt.close('all')
    # exit(1)

    for index in range(args.augmentation + 1):
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
                # send_telegram_picture(a, str(sample['bev_path_filename']))
                send_telegram_picture(a, '')
                plt.close('all')

            # Or use this to debug
            # ---> to find the image inside the dataloader
            #      [(idx, img) for idx, img in enumerate(image_02) if '5232' in img]
            # sample = dataloader.dataset.__getitem__(5302)
            # data = sample['data']
            # label = sample['label']
            # a = plt.figure()
            # plt.imshow(sample['data'] / 255.0)
            # send_telegram_picture(a, str(sample['bev_path_filename']))
            # plt.close('all')

            # for WARPINGS
            # data = sample['data']
            # label = sample['label']
            # a = plt.figure()
            # plt.imshow(sample['data'].squeeze() / 255.0)
            # send_telegram_picture(a, 'fff')
            # plt.close('all')

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
    # parser.add_argument('--rootfolder', default="/media/augusto/500GBHECTOR/augusto/kitti360-augusto", type=str, help='Root folder for all datasets')
    # parser.add_argument('--savefolder', default="/media/augusto/500GBHECTOR/augusto/kitti360-augusto-augmented-warped", type=str, help='Where to save the new data')

    # KITTI-ROAD (kitti2011 relabeled)
    # WARPINGS
    # warpings parser.add_argument('--rootfolder', default="/home/ballardini/DualBiSeNet/KITTI-ROAD/all.txt", type=str, help='Root folder for all datasets')
    # warpings parser.add_argument('--savefolder', default="/home/ballardini/DualBiSeNet/KITTI-ROAD_warped", type=str, help='Where to save the new data')
    # 3D
    # parser.add_argument('--rootfolder', default="/home/ballardini/DualBiSeNet/KITTI-ROAD/all.txt", type=str, help='Root folder for all datasets')
    # parser.add_argument('--savefolder', default="/home/ballardini/DualBiSeNet/KITTI-ROAD_3D", type=str, help='Where to save the new data')

    parser.add_argument('--rootfolder', default="/home/ballardini/DualBiSeNet/KITTI-360/all.txt", type=str, help='Root folder for all datasets')
    parser.add_argument('--savefolder', default="/home/ballardini/DualBiSeNet/KITTI-360_3D", type=str, help='Where to save the new data')


    # ALCALA 26 - AUGUSTO's LAPTOP
    # parser.add_argument('--rootfolder', default="/home/ballardini/Desktop/alcala-26.01.2021_selected/all.txt", type=str, help='Root folder for all datasets')
    # parser.add_argument('--savefolder', default="/home/ballardini/Desktop/alcala-26.01.2021_selected_augmented_warped", type=str, help='Where to save the new data')

    # ALCALA 12 02 2021 000+001
    # parser.add_argument('--rootfolder', default="/home/ballardini/DualBiSeNet/alcala-12.02.2021/001_test_list.txt", type=str, help='Root folder for all datasets')
    # parser.add_argument('--savefolder', default="/home/ballardini/DualBiSeNet/alcala-12.02.2021_augmented_warped_1", type=str, help='Where to save the new data')
    # for the 1-b, remember to set the RANDOM values (ex: random_Rx_degrees and so on...) to ZERO
    # parser.add_argument('--savefolder', default="/home/ballardini/DualBiSeNet/alcala-12.02.2021_augmented_warped_1-b", type=str, help='Where to save the new data')

    parser.add_argument('--execute', type=str, help='Name of the process to execute, see the code')
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

    # if args.telegram:
    #     send_telegram_message("Executing generate.bev.dataset.py")

    try:
        tic = time.time()
        main(args)
        toc = time.time()
        # if args.telegram:
        #     send_telegram_message("Generation of dataset ended correctly after " +
        #                           str(time.strftime("%H:%M:%S", time.gmtime(toc - tic))))

    except (KeyboardInterrupt, SystemExit):
        print("Shutdown requested")
        # if args.telegram:
        #     send_telegram_message("Shutdown requested")
        raise
    except:
        e = sys.exc_info()
        print(e)
        # if args.telegram:
        #     send_telegram_message("Error generating the dataset" + str(e))
