

###
###
###
###     this little script was used to check/develope the sequence dataloader. nothing special here.
###
###     ALSO TO CREATE THE INPUT TO THE KERAS SCRIPT!
###

import os
import numpy as np
import pickle

from dataloaders.sequencedataloader import lstm_txt_dataloader
import torchvision.transforms as transforms
from dataloaders.transforms import GenerateBev, Mirror, Normalize, Rescale, ToTensor
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
#                              folders=['R2_video_0002_camera1_png'])

# dataset = alcala26012021(path_filename='/home/ballardini/Desktop/alcala-26.01.2021/train_list.txt')

# dataset = Sequences_alcala26012021_Dataloader(path_filename='/home/ballardini/Desktop/alcala-26.01.2021/train_list.txt',
#                                               usePIL=False)

# dataset = Sequences_alcala26012021_Dataloader(
#     path_filename='/home/ballardini/DualBiSeNet/alcala-12.02.2021/test_list.txt', usePIL=False)

rgb_image_test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

to_process = [

    ['/tmp/fixedlength6_keras_KITTI-360_warped_train.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_warped_valid.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_warped_test.pickle',
     '/home/ballardini/DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt'],

    ['/tmp/fixedlength6_keras_KITTI-360_3D_train.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_3D_valid.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_3D_test.pickle',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_train_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_test_list.txt'],

    ['/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_train.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_valid.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_test.pickle',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt'],

    ['/tmp/fixedlength6_keras_KITTI-360_train.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_valid.pickle',
     '/tmp/fixedlength6_keras_KITTI-360_test.pickle',
     '/home/ballardini/DualBiSeNet/KITTI-360/train.prefix/prefix_train_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360/validation.prefix/prefix_validation_list.txt',
     '/home/ballardini/DualBiSeNet/KITTI-360/test.prefix/prefix_test_list.txt']

]


# train_filename = '/tmp/ivan_kitti360_warped_train.pickle'
# valid_filename = '/tmp/ivan_kitti360_warped_valid.pickle'
# test_filename = '/tmp/ivan_kitti360_warped_test.pickle'
# train_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt'
# valid_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt'
# test_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt'

for i in to_process:

    train_filename = i[0]
    valid_filename = i[1]
    test_filename = i[2]
    train_path = i[3]
    valid_path = i[4]
    test_path = i[5]

    dataset_train = lstm_txt_dataloader(train_path, transform=rgb_image_test_transforms, all_in_ram=False,
                                        fixed_lenght=2, verbose=False)
    dataset_valid = lstm_txt_dataloader(valid_path, transform=rgb_image_test_transforms, all_in_ram=False,
                                        fixed_lenght=2, verbose=False)
    dataset_test = lstm_txt_dataloader(test_path, transform=rgb_image_test_transforms, all_in_ram=False, fixed_lenght=2,
                                       verbose=False)

    # used to find the number "6" shared with all datasets
    # print('min elements train/val/test: ', dataset_train.min_elements, dataset_valid.min_elements, dataset_test.min_elements)
    # continue

    dataset_train.min_elements = 6
    dataset_valid.min_elements = 6
    dataset_test.min_elements = 6

    train_loader = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False)
    valid_loader = DataLoader(dataset_valid, batch_size=1, num_workers=0, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

    episodes = {}
    for idx, data in enumerate(train_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        print(episodes[idx])
    with open(train_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    episodes = {}
    for idx, data in enumerate(valid_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        print(episodes[idx])
    with open(valid_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    episodes = {}
    for idx, data in enumerate(test_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        print(episodes[idx])
    with open(test_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)


# for idx, data in enumerate(test_loader):
#         print(data['label'], data['path_of_original_images'])

#
#
#
# for idx, data in enumerate(loader):
#         print(idx)

print("End")
