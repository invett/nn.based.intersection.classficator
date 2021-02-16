import numpy as np
import random
import os


# This script takes the some list generated with the labelling script and divided it in three parts to be used in *BOTH*
# RESNET and LSTM.
#
# Initially created to divide KITTI360 in three splits, it can be used to split other datasets as well.
#
#   NEEDED: one txt in the follwing form:
#
#   (base) ballardini@ballardini-T14:~/Desktop/alcala-12.02.2021.000$ head all_frames_labelled_focus_and_c4.txt
#   122302AA/0000000791.png;0
#   122302AA/0000000792.png;0
#   122302AA/0000000793.png;0
#   122302AA/0000000794.png;0
#   122302AA/0000000795.png;0
#   122302AA/0000000796.png;0
#   122302AA/0000000797.png;0
#   122302AA/0000000798.png;0
#   122302AA/0000000799.png;0
#   122302AA/0000000800.png;0
#
#   KITTI360 WARNING! if filename contains '_' then use this line
#
#   current_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[7])
#
#   and create a three files:
#                               prefix_train_list.txt
#                               prefix_validation_list.txt
#                               prefix_test_list.txt

prefix_filename = "prefix_"
prefix_filename = "alcala.12.standard.split."
annotations = []
files = []
iskitti360 = False
overwrite_i_dont_care = False

input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/all_frames_labelled_focus_and_c4.txt'
save_folder = '/tmp/'


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
    file_added = 0
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

            if iskitti360:
                current_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[7])
            else:
                current_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[1])

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
                else:
                    if (frame_class == prev_frame_class) and (current_frame_number == prev_frame_number+1):
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


            file_added = file_added + 1
            current_sequence_frames = current_sequence_frames + 1
            prev_frame_class = frame_class
            prev_frame_number = int(frame_filename.replace('_', '.').replace('/', '.').split('.')[1])
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
                print("Frames skipped are: ")
                print(sq)
                excluded_counter = excluded_counter + len(sq)

    type_x_sequences = type_x_sequences_.copy()

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

    # save the lists using the save_folder as root
    train_filename = prefix_filename + 'train_list.txt'
    validation_filename = prefix_filename + 'validation_list.txt'
    test_filename = prefix_filename + 'test_list.txt'

    if not overwrite_i_dont_care:
        assert not os.path.isfile(os.path.join(save_folder, train_filename)), 'File exists: ' + os.path.join(save_folder, train_filename)
        assert not os.path.isfile(os.path.join(save_folder, validation_filename)), 'File exists' + os.path.join(save_folder, validation_filename)
        assert not os.path.isfile(os.path.join(save_folder, test_filename)), 'File exists' + os.path.join(save_folder, test_filename)

    with open(os.path.join(save_folder, train_filename), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(save_folder, validation_filename), 'w') as f:
        for item in validation_list:
            f.write("%s\n" % item)
    with open(os.path.join(save_folder, test_filename), 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)

    print("Finish")

    print("Maybe you want to create the symbolic links....")

    print('while read line; do folder=$(echo $line | cut -d \'/\' -f 1); filenamewithpath=$(echo $line | cut --d \';\' -f 1); filename=$(echo $filenamewithpath | cut --d \'/\' -f 2); echo mkdir -p test/$folder; echo ln -s ../../$filenamewithpath test/$folder/$filename ; done < test_list.txt')


with open(input_file, "r") as f:
    all_lines = f.read().splitlines()

for line in all_lines:
    file, label = line.split(';')
    files.append(file)
    annotations.append(int(label))

files = [files]
annotations = [annotations]  # stupid line to make previous code work here

summary(annotations, files)
