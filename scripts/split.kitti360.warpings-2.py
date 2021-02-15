import numpy as np
import random
import os

def summary(annotations, files):
    print("Computing annotations...")
    type_0 = sum(sum(i == 0 for i in j) for j in annotations)
    type_1 = sum(sum(i == 1 for i in j) for j in annotations)
    type_2 = sum(sum(i == 2 for i in j) for j in annotations)
    type_3 = sum(sum(i == 3 for i in j) for j in annotations)
    type_4 = sum(sum(i == 4 for i in j) for j in annotations)
    type_5 = sum(sum(i == 5 for i in j) for j in annotations)
    type_6 = sum(sum(i == 6 for i in j) for j in annotations)
    type_x_frames = type_0 + type_1 + type_2 + type_3 + type_4 + type_5 + type_6

    type_x_sequences = [[], [], [], [], [], [], []]

    sequences = 0
    sequences_frame_number = []
    for i in range(1):
        current_sequence_frames = 0
        prev_frame_class = None
        prev_frame_number = None

        current_sequence_filenames = []

        # iterate both annotations and filename lists together, contains labels and frame-names
        for frame_class, frame_filename in zip(annotations[i], files[i]):
            print(frame_filename)
            if current_sequence_frames == 0 and frame_class == -1:
                continue

            current_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[7])
            if prev_frame_number is None:
                prev_frame_number = current_frame_number - 1

            # check for sequence. if the current frame number is not the previous+1, then we have a new sequence.
            # we need to enter here also if we're analyzing the last frame, or we will lost ALL the frames in the last
            # sequence of the folder
            if not (prev_frame_class is None or (frame_class == prev_frame_class)) or (frame_filename == files[i][-1])\
                    or (current_frame_number != prev_frame_number+1):

                # if we're in the last frame, we need to re-check if the last frame belongs to the sequence again
                if frame_filename == files[i][-1]:
                    if frame_class == prev_frame_class:
                        current_sequence_frames = current_sequence_frames + 1
                        current_sequence_filenames.append(frame_filename)

                if prev_frame_class == 0:
                    type_x_sequences[0].append(current_sequence_filenames.copy())
                if prev_frame_class == 1:
                    type_x_sequences[1].append(current_sequence_filenames.copy())
                if prev_frame_class == 2:
                    type_x_sequences[2].append(current_sequence_filenames.copy())
                if prev_frame_class == 3:
                    type_x_sequences[3].append(current_sequence_filenames.copy())
                if prev_frame_class == 4:
                    type_x_sequences[4].append(current_sequence_filenames.copy())
                if prev_frame_class == 5:
                    type_x_sequences[5].append(current_sequence_filenames.copy())
                if prev_frame_class == 6:
                    type_x_sequences[6].append(current_sequence_filenames.copy())

                if len(current_sequence_filenames) != current_sequence_frames:
                    print("Error")
                sequences_frame_number.append(current_sequence_frames)
                prev_frame_class = None
                prev_frame_number = None
                current_sequence_frames = 0
                sequences += 1
                current_sequence_filenames = []
                continue
            current_sequence_frames = current_sequence_frames + 1
            prev_frame_class = frame_class
            prev_frame_number = int(frame_filename.replace('_','.').replace('/','.').split('.')[7])
            current_sequence_filenames.append(frame_filename)

    # some sequences of kitti 360 have only one frame, others 2 or 3 ... we need to prune these sequences
    threshold = 5
    excluded_counter = 0
    type_x_sequences_ = [[], [], [], [], [], [], []]
    for index, val in enumerate(type_x_sequences):
        for sq in val:
            if len(sq) > threshold:
                type_x_sequences_[index].append(sq)
            else:
                print("Skipping sequence with #frames: " + str(len(sq)))
                excluded_counter = excluded_counter + len(sq)



    print("Type0 seq/frames:", len(type_x_sequences[0]), "/", type_0, "\t-\t", [len(i) for i in type_x_sequences[0]])
    print("Type1 seq/frames:", len(type_x_sequences[1]), "/", type_1, "\t-\t", [len(i) for i in type_x_sequences[1]])
    print("Type2 seq/frames:", len(type_x_sequences[2]), "/", type_2, "\t-\t", [len(i) for i in type_x_sequences[2]])
    print("Type3 seq/frames:", len(type_x_sequences[3]), "/", type_3, "\t-\t", [len(i) for i in type_x_sequences[3]])
    print("Type4 seq/frames:", len(type_x_sequences[4]), "/", type_4, "\t-\t", [len(i) for i in type_x_sequences[4]])
    print("Type5 seq/frames:", len(type_x_sequences[5]), "/", type_5, "\t-\t", [len(i) for i in type_x_sequences[5]])
    print("Type6 seq/frames:", len(type_x_sequences[6]), "/", type_6, "\t-\t", [len(i) for i in type_x_sequences[6]])
    print("Overall Sequences: ", sequences)
    print("The number of frames associated to each sequence is: ", sequences_frame_number)
    print("Sum of frames in seq ", sum(sequences_frame_number), '/', type_x_frames)

    train_list = []
    validation_list = []
    test_list = []


    type_x_sequences = type_x_sequences_.copy()

    for i in range(7):
        # split_train_val_test = list(range(len(type_x_sequences[i])))
        # random.shuffle(split_train_val_test)
        # a = split_train_val_test[:int(len(split_train_val_test) * 0.8)]
        # a = split_train_val_test[:int(len(split_train_val_test) * 0.8)]
        # a = split_train_val_test[:int(len(split_train_val_test) * 0.8)]

        # create a copy of the sequence_x list. Then shuffle, then split using np.split
        # in three parts 0.7 | 0.2 | 0.1 for train/validation/test


        tosplit = type_x_sequences[i].copy()
        random.shuffle(tosplit)
        split_train_val_test = np.split(tosplit, [int(len(tosplit) * 0.7), int(len(tosplit) * 0.9)])

        # and append those lists for each of the train/val/test sets ; don't forget to add also the label at the end
        # so the file will have namefile;label

        base_folder = ''

        for split_train in split_train_val_test[0]:
            for filename in split_train:
                towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                train_list.append(towrite + ';' + str(i))
        for split_valid in split_train_val_test[1]:
            for filename in split_valid:
                towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                validation_list.append(towrite + ';' + str(i))
        for split_test in split_train_val_test[2]:
            for filename in split_test:
                towrite = os.path.relpath(filename, os.path.commonpath([base_folder, filename]))
                test_list.append(towrite + ';' + str(i))

    print("Frames for Train/Val/Test: ", len(train_list), "/", len(validation_list), "/", len(test_list), "\tTot: ",
          len(train_list) + len(validation_list) + len(test_list))

    base_folder = '/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/'

    # save the lists using the base_folder as root
    # WERE train_list.txt   validation_list.txt     test_list.txt
    with open(os.path.join(base_folder, 'train_list_2nd.split.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(base_folder, 'validation_list_2nd.split.txt'), 'w') as f:
        for item in validation_list:
            f.write("%s\n" % item)
    with open(os.path.join(base_folder, 'test_list_2nd.split.txt'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)

    print("Finish")


# annotations = list(np.loadtxt('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/all_annotations.txt', dtype='str'))
# files = list(np.loadtxt('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/all_files.txt', dtype='str'))

annotations = []
files = []

# with open('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/all_files_2nd.split.txt', "r") as f:
#     files = [f.read().splitlines()]
# with open('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/all_annotations_2nd.split.txt', "r") as a:
#     annotations = [a.read().splitlines()]

with open('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/validation/bugged/all_files_bugged.txt', "r") as f:
    files = [f.read().splitlines()]
with open('/home/ballardini/DualBiSeNet/kitti360-augusto-warping/kitti360-augusto-warping_flat/validation/bugged/all_annotations_bugged.txt', "r") as a:
    annotations = [a.read().splitlines()]

for i in range(0, len(annotations[0])):
    annotations[0][i] = int(annotations[0][i])

annotations = [np.asarray(annotations[0])]

summary(annotations, files)
