import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import sys
import time

import torchvision.transforms as transforms
from dataloaders.transforms import Rescale, ToTensor, Normalize, GenerateBev, Mirror, GenerateNewDataset, \
    WriteDebugInfoOnNewDataset
from dataloaders.sequencedataloader import fromAANETandDualBisenet, teacher_tripletloss, teacher_tripletloss_generated

from miscellaneous.utils import send_telegram_message
from miscellaneous.utils import send_telegram_picture
import matplotlib.pyplot as plt
import torch


# This script allows for evaluating the GT (frames_topology.txt files) with respect the OSM files.
# execute this script then:
#
#   1. we'll send a message over telegram with the detailed information (you'll visually check if  the ground truth
#   corresponds)
#   2. save this info in a folder (hard-coded here in the code)

def main(args):
    folders = np.array([os.path.join(args.rootfolder, folder) for folder in os.listdir(args.rootfolder) if
                        os.path.isdir(os.path.join(args.rootfolder, folder))])

    dataset = teacher_tripletloss(folders, args.distance, transform=[])
    # dataset = teacher_tripletloss_generated(elements=3, rnd_width=2.0, rnd_angle=0.4, rnd_spatial=9.0, noise=True,
    # transform=[])

    # num_workers starts from 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    i = 0
    for sample in dataloader:
        print(str(i) + "/" + str(len(dataloader)))
        i += 1

        emptyspace = 255 * torch.ones([300, 30, 3], dtype=torch.uint8)
        a = plt.figure()
        plt.imshow(torch.cat((sample['anchor'].squeeze(), emptyspace, sample['positive'].squeeze(), emptyspace,
                              sample['negative'].squeeze(), emptyspace, sample['ground_truth_image'].squeeze()), 1))

        if args.savefile:
            image_name = args.savefolder +  \
                         str(sample['filename_anchor'][0]).split(sep="/")[6] + "/" + os.path.basename(
                    str(sample['filename_anchor'][0]))
            text_name = str.replace(image_name, ".png", ".txt")

            plt.savefig(image_name)

            textfile = open(text_name, "w")
            textfile.write(
                str(sample['filename_anchor']) + " is type: " + str(sample['label_anchor'].numpy()[0]) + "\n" + str(
                    sample['filename_positive']) + " is type: " + str(sample['label_positive'].numpy()[0]) + "\n" + str(
                    sample['filename_negative']) + " is type: " + str(
                    sample['label_negative'].numpy()[0]) + "\n\nLast IMG is the GT of the ANCHOR\n" +
                "anchor lat: " + str(sample['anchor_oxts_lat'][0]) + "\n" +
                "anchor lon: " + str(sample['anchor_oxts_lon'][0]) + "\n" +
                "positive lat: " + str(sample['positive_oxts_lat'][0]) + "\n" +
                "positive lon: " + str(sample['positive_oxts_lon'][0]) + "\n" +
                "negative lat: " + str(sample['negative_oxts_lat'][0]) + "\n" +
                "negative lon: " + str(sample['negative_oxts_lon'][0]))
            textfile.close()

        if args.telegram:
            send_telegram_picture(a, str(sample['filename_anchor']) + " is type: " + str(
                sample['label_anchor'].numpy()[0]) + "\n" + str(sample['filename_positive']) + " is type: " + str(
                sample['label_positive'].numpy()[0]) + "\n" + str(sample['filename_negative']) + " is type: " + str(
                sample['label_negative'].numpy()[0]) + "\n\nLast IMG is the GT of the ANCHOR\n" +
                                  "anchor lat: " + str(sample['anchor_oxts_lat'][0]) + "\n" +
                                  "anchor lon: " + str(sample['anchor_oxts_lon'][0]) + "\n" +
                                  "positive lat: " + str(sample['positive_oxts_lat'][0]) + "\n" +
                                  "positive lon: " + str(sample['positive_oxts_lon'][0]) + "\n" +
                                  "negative lat: " + str(sample['negative_oxts_lat'][0]) + "\n" +
                                  "negative lon: " + str(sample['negative_oxts_lon'][0]))

        plt.close('all')

    print("End.")

    # for index in range(args.augmentation):
    #     print("Generating run {} ... ".format(index))
    #     if args.telegram:
    #         send_telegram_message("Generating run {} ... ".format(index))
    #
    #     # RESET SEED to have np.random working correctly https://github.com/pytorch/pytorch/issues/5059
    #     np.random.seed()
    #
    #     for sample in dataloader:
    #         data = sample['data']
    #         label = sample['label']
    #         print(sample['bev_path_filename'])
    #         #break
    #
    #     print("Run {} generated".format(index))
    #     if args.telegram:
    #         send_telegram_message("Run {} generated successfully".format(index))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--rootfolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw", type=str,
                        help='Root folder for all datasets')
    parser.add_argument('--savefolder',
                        default="/media/augusto/500GBDISK/nn.based.intersection.classficator.data/check_osm_again/", type=str,
                        help='Where to save the new data')
    parser.add_argument('--augmentation', type=int, default=50, help='How many files generate for each of the BEVs')
    parser.add_argument('--workers', type=int, default=0, help='How many workers for the dataloader')
    parser.add_argument('--telegram', action='store_true', help='Send info through Telegram')
    parser.add_argument('--savefile', action='store_true', help='Send info through Telegram')
    parser.add_argument('--debug', action='store_true', help='Print filenames as walking the filesystem')
    parser.add_argument('--distance', type=float, default=20.0, help='Distance from the cross')

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
