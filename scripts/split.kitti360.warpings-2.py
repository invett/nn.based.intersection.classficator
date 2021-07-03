"""
    from a file containing sequential filenames, create a split train/val/test

    input file must contain something in this form

    122302AA/0000000791.png;0
    122302AA/0000000792.png;0
    ....
    122302AA/0000000800.png;0
    122302AA/0000004234.png;0

    the split of the last part of the name is done inside 'split_dataset' using tokens.

    HEY!!!  if don't have the file, you can CREATE THE FILE using create_unique_file_from_GAN_generated_folder.py !!!!

"""

from miscellaneous.utils import split_dataset

# Parameters of this script
annotations = []
files = []
input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/120445AA.122302AA.164002AA.165810AA.txt'
# input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/120445AA.122302AA.txt'
# input_file = '/home/ballardini/Desktop/alcala-12.02.2021.000/164002AA.165810AA.txt'

# parameters that will be passed to split_dataset
prefix_filename = "seq.120445AA.122302AA.164002AA.165810AA."
# prefix_filename = "seq.120445AA.122302AA."
# prefix_filename = "seq.164002AA.165810AA."
overwrite_i_dont_care = False
save_folder = '/tmp/'

with open(input_file, "r") as f:
    all_lines = f.read().splitlines()

for line in all_lines:
    try:
        file, label = line.split(';')
        files.append(file)
        annotations.append(int(label))
    except:
        print(line)
        exit(1)

# split_dataset should ??? :) work with multiple folders for the labelling script so annotations should be a list of
# lists...
files = [files]
annotations = [annotations]

split_dataset(annotations=annotations, files=files, prefix_filename=prefix_filename, save_folder=save_folder,
              overwrite_i_dont_care=overwrite_i_dont_care, extract_field_from_path=2)
