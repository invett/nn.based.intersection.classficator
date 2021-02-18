"""
    from a file containing sequential filenames, create a split train/val/test

    input file must contain something in this form

    122302AA/0000000791.png;0
    122302AA/0000000792.png;0
    ....
    122302AA/0000000800.png;0
    122302AA/0000004234.png;0

    the split of the last part of the name is done inside 'split_dataset' using tokens.
"""

from miscellaneous.utils import split_dataset

# Parameters of this script
annotations = []
files = []
input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/all_frames_labelled_focus_and_c4.txt'

# parameters that will be passed to split_dataset
prefix_filename = "alcala.12.standard.split."
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

split_dataset(annotations, files, prefix_filename, save_folder, overwrite_i_dont_care, extract_field_from_path=1)
