import csv
import os
import numpy as np

root = '../../DualBiSeNet/data_raw/'

folders = np.array([os.path.join(root, folder) for folder in sorted(os.listdir(root)) if
                    os.path.isdir(os.path.join(root, folder))])

for folder in folders:
    file = os.path.join(folder, 'frams_topology.txt')
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            print(', '.join(row))

