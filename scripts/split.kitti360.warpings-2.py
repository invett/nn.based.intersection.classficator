import numpy as np
import random
import os
from miscellaneous.utils import split_dataset

# Parameters of this script
annotations = []
files = []
input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/all_frames_labelled_focus_and_c4.txt'

# parameters that will be passed to split_dataset
prefix_filename = "prefix_"
prefix_filename = "alcala.12.standard.split."
iskitti360 = False
overwrite_i_dont_care = False
save_folder = '/tmp/'

with open(input_file, "r") as f:
    all_lines = f.read().splitlines()

for line in all_lines:
    file, label = line.split(';')
    files.append(file)
    annotations.append(int(label))

# split_dataset should ??? :) work with multiple folders for the labelling script so annotations should be a list of
# lists...
files = [files]
annotations = [annotations]

split_dataset(annotations, files, prefix_filename, save_folder, iskitti360, overwrite_i_dont_care)
