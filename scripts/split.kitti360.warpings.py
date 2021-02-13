import numpy as np
import random
import os

split_train_val_test = list(np.loadtxt('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping/all.txt', dtype='str'))

random.shuffle(split_train_val_test)

split = np.split(split_train_val_test, [int(len(split_train_val_test) * 0.7), int(len(split_train_val_test) * 0.9)])

train_list = split[0].tolist()
validation_list = split[1].tolist()
test_list = split[2].tolist()

base_folder = '.'

# save the lists using the base_folder as root
with open(os.path.join(base_folder, 'train_list.txt'), 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)
with open(os.path.join(base_folder, 'validation_list.txt'), 'w') as f:
    for item in validation_list:
        f.write("%s\n" % item)
with open(os.path.join(base_folder, 'test_list.txt'), 'w') as f:
    for item in test_list:
        f.write("%s\n" % item)
