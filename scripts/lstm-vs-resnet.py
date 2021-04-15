# comparison beweeen resnet and lstm

# *** RESNET vs LSTM ***
#
# 1. naive:
#     a. per-frame classification with resnet/cnn
#     b. per-secuence classification with lstm
#
# 2. consider sequences also with resnet/cnn, and
#     a. take the max(sequence) (max number of classifications, ie with 5 frames, 1 1 2 1 2 = 1) as output/cnn, to
#        compare with the final output/lstm.
#     b. per-frame comparison between resnet/cnn and lstm; in this way we can assess the temporal effictiveness of
#        the lstm temporal integration
#         resnet 1
#         lstm   1            -->  compare
#         ---
#         resnet x 1
#         lstm   1 1          -->  compare
#         ---
#         resnet x x 2
#         lstm   1 1 2        -->  compare
#         ---
#         resnet x x x 1
#         lstm   1 1 2 1      -->  compare
#         ---
#         resnet x x x x 2
#         lstm   1 1 2 1 1    -->  compare
#         ---
#                             min/max/average?
#
# 3. put more images in CNN
#     a. more channels, from Bx3xHxW to Bx3*NxHxW
#     b. 3D convs: from BxCxHxW to BxCxDxHxW
#
# 4. build a temporal integration filter upon the results on resnet/cnn. but in this case the comparison becomes
#    filter vs lstm...

import csv
import os
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support
from collections import Counter
import numpy as np

base = '/home/ballardini/workspace/nn.based.intersection.classficator/wiki/resnet-vs-lstm/kitti360_warped/'
resnet = 'prefix_test_list_resnet_export_svm.txt'
lstm = 'prefix_test_list_lstm_export_svm.txt'


def dicttolist(dict_):
    dictlist = []
    for key, value in dict_.items():
        dictlist.append([int(i) for i in value])
    return dictlist


def createlist(num, value):
    lst = []
    for i in range(num):
        lst.append(value)
    return lst


def evaluate(base, file, what):
    filelist = []
    GT = []
    prediction = []
    GT_persequence = []

    with open(os.path.join(base, file), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='\\')
        for row in spamreader:
            # print(', '.join(row))
            filelist.append(row[0])
            GT.append(row[1])
            prediction.append(row[2])

    seq_dict = {}
    seq_dict_labels = {}
    sq = 0
    sequence_filename = []
    sequence_label_prediction = []
    prev_framenumber = None
    for file, GT_i, prediction_i in zip(filelist, GT, prediction):
        # take into account also the 'warpings' folder that contains appendix numbers like
        # framenumber.data-augmentation-counter.png like 0000000084.001.png
        # the following trick does the job, but it doesn't work for folders with multiple
        # data augmentation files, ie, filename.002+.png won't be handled correctly.
        # TODO: improve the creation of sequences from data-augmented folders.

        if '2013_' in file:
            # BRUTAL kitti360 patch: ... the frame numbering is different since they contain "folder" in the
            # filename itself... i'll try to catch this searching '2013_' in the string ...
            # 6/2013_05_28_drive_0009_sync_0000007162.001.png
            frame_number = int(os.path.splitext(os.path.split(file)[1])[0].split('.')[0].split('_')[-1])
        else:
            frame_number = int(os.path.splitext(os.path.split(file)[1])[0].split('.')[0])

        # check for sequence. if the current frame number is not the previous+1, then we have a new sequence.
        if not (prev_framenumber is None or (frame_number == (prev_framenumber + 1))):
            seq_dict[sq] = sequence_filename.copy()
            seq_dict_labels[sq] = sequence_label_prediction.copy()
            GT_persequence.append(prev_GT)
            sequence_filename.clear()
            sequence_label_prediction.clear()
            sq += 1

        sequence_filename.append(file)
        sequence_label_prediction.append(prediction_i)
        prev_framenumber = frame_number
        prev_GT = GT_i

    # check if we have something that need to add
    if sequence_filename:
        seq_dict[sq] = sequence_filename.copy()
        seq_dict_labels[sq] = sequence_label_prediction.copy()
        GT_persequence.append(prev_GT)
        sequence_filename.clear()
        sequence_label_prediction.clear()
        sq += 1

    # naive_last_element_of_sequence = [seq_dict_labels[i][-1] for i in seq_dict_labels]


    persequence_max_accuracy_sequences = []
    naive_accuracy_sequences = []
    for seq in seq_dict:
        GT_ = []
        prediction_ = []
        for img in seq_dict[seq]:
            index = filelist.index(img)
            GT_.append(GT[index])
            prediction_.append(prediction[index])
        value = accuracy_score(GT_, prediction_)
        naive_accuracy_sequences.append(value)

        # get the max number of occurrencies in the sequence and then calculate the accuracy
        max_occurrences_value = max(prediction_, key=prediction_.count)
        persequence_max_accuracy_sequences.append(max_occurrences_value)

    #naive_accuracy_sequences = np.sum(np.array(naive_accuracy_sequences)) / len(seq_dict)
    #persequence_max_accuracy_sequences = accuracy_score(GT_persequence, persequence_max_accuracy_sequences)

    return filelist, GT, prediction, GT_persequence, seq_dict, seq_dict_labels, persequence_max_accuracy_sequences


def naive_last_element_of_sequence(lstm_seq_dict_labels_):
    return [lstm_seq_dict_labels_[i][-1] for i in lstm_seq_dict_labels_]


def improved_b(seq_dict_labels, GT_persequence_full):
    acc_score_list = []
    for pred, gt in zip(dicttolist(seq_dict_labels), GT_persequence_full):
        #print(pred)
        #print(gt)
        acc_score_list.append(accuracy_score(gt, pred))
        #for resnet_element, gt_element in zip(resnet_seq, gt_seq):
        #    print(resnet_element)
    return np.mean(np.array(acc_score_list))


resnet_filelist, resnet_GT, resnet_prediction, resnet_GT_persequence, resnet_seq_dict, resnet_seq_dict_labels, resnet_persequence_max_accuracy_sequences = evaluate(base, resnet, 'resnet')
lstm_filelist, lstm_GT, lstm_prediction, lstm_GT_persequence,  lstm_seq_dict, lstm_seq_dict_labels, lstm_persequence_max_accuracy_sequences = evaluate(base, lstm, 'lstm')

# sanity check
if not (resnet_GT == lstm_GT):
    exit(-1)

# *** SUPPORT VALUES ***
resnet_GT_persequence_full = [createlist(len(resnet_seq_dict[int(i)]), int(resnet_GT_persequence[i])) for i in range(len(resnet_GT_persequence))]
lstm_GT_persequence_full = [createlist(len(lstm_seq_dict[int(i)]), int(lstm_GT_persequence[i])) for i in range(len(lstm_GT_persequence))]

# *** CREATE METRICS ***
# (1a) per frame classification, resnet (all frames)
# (1b) per sequence classification, lstm (output after all frames)
# (2a) per sequence mode, value that appears most often in each sequence, resnet
# (2b-1) resnet
# (2b-2) lstm

naive_resnet = accuracy_score(resnet_GT, resnet_prediction)
naive_lstm = accuracy_score(lstm_GT_persequence, naive_last_element_of_sequence(lstm_seq_dict_labels))
improved_resnet_sequence = accuracy_score(resnet_GT_persequence, resnet_persequence_max_accuracy_sequences)
imp2b1 = improved_b(resnet_seq_dict_labels, resnet_GT_persequence_full)
imp2b2 = improved_b(lstm_seq_dict_labels, lstm_GT_persequence_full)

print('RESNET Naive per-frame    : \t' + str(naive_resnet))
print('LSTM   Naive per-sequence : \t' + str(naive_lstm))
print('RESNET max-in-sequence    : \t' + str(improved_resnet_sequence))
print('RESNET improved 2b1       : \t' + str(imp2b1))
print('LSTM   improved 2b2       : \t' + str(imp2b2))


print(precision_recall_fscore_support(resnet_GT, resnet_prediction, average='micro'))
print(precision_recall_fscore_support(lstm_GT_persequence, naive_last_element_of_sequence(lstm_seq_dict_labels), average='micro'))
print(precision_recall_fscore_support(resnet_GT_persequence, resnet_persequence_max_accuracy_sequences, average='micro'))

# print('perseq-a , max per-sequence: \t' + str(persequence_a_accuracy_sequences))
