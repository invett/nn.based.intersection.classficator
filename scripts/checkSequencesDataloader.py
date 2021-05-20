

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
import shutil

from dataloaders.sequencedataloader import lstm_txt_dataloader
from dataloaders.sequencedataloader import txt_dataloader
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

    # ['/tmp/fixedlength6_keras_KITTI-360_warped_train.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_warped_valid.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_warped_test.pickle',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt',
    #  'KITTI-360_warped'],
    #
    # ['/tmp/fixedlength6_keras_KITTI-360_3D_train.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_3D_valid.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_3D_test.pickle',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_train_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_validation_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D/prefix_test_list.txt',
    #  'KITTI-360_3D'],
    #
    # ['/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_train.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_valid.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_3D-MASKED_test.pickle',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_train_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_validation_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360_3D-MASKED/prefix_test_list.txt',
    #  'KITTI-360_3D-MASKED']
    #
    # ['/tmp/fixedlength6_keras_KITTI-360_train.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_valid.pickle',
    #  '/tmp/fixedlength6_keras_KITTI-360_test.pickle',
    #  '/home/ballardini/DualBiSeNet/KITTI-360/train.prefix/prefix_train_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360/validation.prefix/prefix_validation_list.txt',
    #  '/home/ballardini/DualBiSeNet/KITTI-360/test.prefix/prefix_test_list.txt',
    #  'KITTI-360']

    ['/tmp/fixedlength6_keras_alcala26_train.pickle',
     '/tmp/fixedlength6_keras_alcala26_valid.pickle',
     '/tmp/fixedlength6_keras_alcala26_test.pickle',
     '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_train_list.txt',
     '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_validation_list.txt',
     '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_test_list.txt',
     'alcala26-15frame']

]

txtdataset = txt_dataloader('/media/14TBDISK/ballardini/imagenet/prefix_all_unlabeled.txt',
                               transform=rgb_image_test_transforms)
txtdataloader = DataLoader(txtdataset, batch_size=1, num_workers=1, shuffle=False)



# train_filename = '/tmp/ivan_kitti360_warped_train.pickle'
# valid_filename = '/tmp/ivan_kitti360_warped_valid.pickle'
# test_filename = '/tmp/ivan_kitti360_warped_test.pickle'
# train_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/train.prefix/prefix_train_list.txt'
# valid_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/validation.prefix/prefix_validation_list.txt'
# test_path = '/home/ballardini/DualBiSeNet/KITTI-360_warped/test.prefix/prefix_test_list.txt'
# name = 'qualcosa'

for i in to_process:

    train_filename = i[0]
    valid_filename = i[1]
    test_filename = i[2]
    train_path = i[3]
    valid_path = i[4]
    test_path = i[5]
    process_name = i[6]

    # HERE!!!! HERE!!! Here what?! Specify here if you want a specific "sequence lenght" according to the policies
    dataset_train = lstm_txt_dataloader(train_path, transform=rgb_image_test_transforms, all_in_ram=False,
                                        fixed_lenght=2, verbose=False)
    dataset_valid = lstm_txt_dataloader(valid_path, transform=rgb_image_test_transforms, all_in_ram=False,
                                        fixed_lenght=2, verbose=False)
    dataset_test = lstm_txt_dataloader(test_path, transform=rgb_image_test_transforms, all_in_ram=False,
                                       fixed_lenght=2, verbose=False)

    # AND .... READ HERE!!! >>>> used to find the number "6" shared with all datasets (MAGIC NUMBER)
    print('min elements train/val/test: ', dataset_train.min_elements, dataset_valid.min_elements, dataset_test.min_elements)
    exit(-21)
    continue

    dataset_train.min_elements = 15  # <<<<<< READ ABOVE: HOW TO FIND THIS MAGIC NUMBER
    dataset_valid.min_elements = 15  # <<<<<< READ ABOVE: HOW TO FIND THIS MAGIC NUMBER
    dataset_test.min_elements = 15   # <<<<<< READ ABOVE: HOW TO FIND THIS MAGIC NUMBER

    train_loader = DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False)
    valid_loader = DataLoader(dataset_valid, batch_size=1, num_workers=1, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False)

    episodes = {}
    for idx, data in enumerate(train_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        # print(episodes[idx])
        print('train dataset', idx)
    with open(train_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ROOT_PATH = os.path.join('/tmp/pytorchvideo', process_name)
    os.makedirs(ROOT_PATH)
    annotation_file = open(os.path.join(ROOT_PATH, 'annotations_train.txt'), 'w+')

    COMPOSED_BASE_PATH = os.path.join('/tmp/pytorchvideo', process_name, 'train')
    crossing_type = 'n'
    episode_id = 0
    for episode in episodes:
        episode_ = episodes[episode]
        # if crossing_type != episode_['gt']:
        #     episode_id = 0
        crossing_type = episode_['gt']
        destination_path = os.path.join(COMPOSED_BASE_PATH, crossing_type, str(episode_id))
        episode_id = episode_id + 1
        os.makedirs(destination_path)
        for index, frame in enumerate(episode_['frames']):
            input = frame[0]
            output = os.path.join(destination_path, 'img_' + str(index).zfill(3) + '.png')
            shutil.copy2(input, output)
            #line = os.path.split(os.path.relpath(output, ROOT_PATH))[0] + ' ' + '0' + ' ' + str(
            #       dataset_train.min_elements - 1) + ' ' + episode_['gt'] + '\n'
            line = 'ovid_' + str(episode_id) + ' ' + str(episode_id) + ' ' + str(index) + ' ' + os.path.relpath(output,
                                                                                                      COMPOSED_BASE_PATH) + ' ' + \
                   '\"' + episode_['gt'] + '\"\n'
            annotation_file.write(line)
    annotation_file.close()

    # ------------------------------------------------------------------------------------------------------------------
    annotation_file = open(os.path.join(ROOT_PATH, 'annotations_validation.txt'), 'w+')
    episodes = {}
    for idx, data in enumerate(valid_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        # print(episodes[idx])
        print('val dataset', idx)
    with open(valid_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    COMPOSED_BASE_PATH = os.path.join('/tmp/pytorchvideo', process_name, 'validation')
    crossing_type = 'n'
    episode_id = 0
    for episode in episodes:
        episode_ = episodes[episode]
        # if crossing_type != episode_['gt']:
        #     episode_id = 0
        crossing_type = episode_['gt']
        destination_path = os.path.join(COMPOSED_BASE_PATH, crossing_type, str(episode_id))
        episode_id = episode_id + 1
        os.makedirs(destination_path)
        for index, frame in enumerate(episode_['frames']):
            input = frame[0]
            output = os.path.join(destination_path, 'img_' + str(index).zfill(3) + '.png')
            shutil.copy2(input, output)
            #line = os.path.split(os.path.relpath(output, ROOT_PATH))[0] + ' ' + '0' + ' ' + str(
            #       dataset_train.min_elements - 1) + ' ' + episode_['gt'] + '\n'
            line = 'ovid_' + str(episode_id) + ' ' + str(episode_id) + ' ' + str(index) + ' ' + os.path.relpath(output,
                                                                                                      COMPOSED_BASE_PATH) + ' ' + \
                   '\"' + episode_['gt'] + '\"\n'
            annotation_file.write(line)
    annotation_file.close()

    # ------------------------------------------------------------------------------------------------------------------
    annotation_file = open(os.path.join(ROOT_PATH, 'annotations_test.txt'), 'w+')
    episodes = {}
    for idx, data in enumerate(test_loader):
        episodes[idx] = {'id': idx, 'gt': data['label'][0], 'frames': data['path_of_original_images']}
        # print(episodes[idx])
        print('test dataset', idx)
    with open(test_filename, 'wb') as handle:
        pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    COMPOSED_BASE_PATH = os.path.join('/tmp/pytorchvideo', process_name, 'test')
    crossing_type = 'n'
    episode_id = 0
    for episode in episodes:
        episode_ = episodes[episode]
        # if crossing_type != episode_['gt']:
        #     episode_id = 0
        crossing_type = episode_['gt']
        destination_path = os.path.join(COMPOSED_BASE_PATH, crossing_type, str(episode_id))
        episode_id = episode_id + 1
        os.makedirs(destination_path)
        for index, frame in enumerate(episode_['frames']):
            input = frame[0]
            output = os.path.join(destination_path, 'img_' + str(index).zfill(3) + '.png')
            shutil.copy2(input, output)
            #line = os.path.split(os.path.relpath(output, ROOT_PATH))[0] + ' ' + '0' + ' ' + str(
            #       dataset_train.min_elements - 1) + ' ' + episode_['gt'] + '\n'
            line = 'ovid_' + str(episode_id) + ' ' + str(episode_id) + ' ' + str(index) + ' ' + os.path.relpath(output,
                                                                                                      COMPOSED_BASE_PATH) + ' ' + \
                   '\"' + episode_['gt'] + '\"\n'
            annotation_file.write(line)
    annotation_file.close()


# for idx, data in enumerate(test_loader):
#         print(data['label'], data['path_of_original_images'])

#
#
#
# for idx, data in enumerate(loader):
#         print(idx)

print("End")
