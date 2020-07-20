import os
import sys
import cv2
import argparse
import warnings

def main(args):

    for root, subdirs, files in os.walk(args.rootfolder):
        print(subdirs)
        for name in subdirs:
            folder = os.path.join(args.rootfolder, name, args.maskfolder)
            for filename in os.listdir(folder):
                filename = os.path.join(folder, filename)

                alvaromask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

                if args.debug:
                    print("Checking ", filename)

                if (alvaromask > 0).sum() < args.threshold:
                    print(filename)
        break




if __name__ == '__main__':
    # basic parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootfolder', default="/home/malvaro/Documentos/DualBiSeNet/data_raw", type=str, help='Root folder for all datasets')
    parser.add_argument('--maskfolder', default="alvaromask", type=str, help='Which folder to check inside the root folder')
    parser.add_argument('--threshold', default=1100, type=int, help='List files with less than THRESHOLD ones in the image')
    parser.add_argument('--debug', default=False, type=bool, help='Print namefiles as walking the filesystem')

    # the value 1100 corresponds to the last frames after the bugged 1788 which didn't contain 1 single element... all
    # consecutive images are then discarded
    # /home/malvaro/Documentos/DualBiSeNet/data_raw/2011_10_03_drive_0034_sync/alvaromask/0000001790pred.png

    args = parser.parse_args()

    print(args)
    warnings.filterwarnings("ignore")
    main(args)