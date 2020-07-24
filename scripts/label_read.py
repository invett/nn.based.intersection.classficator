import os
import pandas as pd

path = '../../DualBiSeNet/data_raw/'

with open('labels.txt', 'w') as logfile:
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == 'frames_topology.txt':
                gt_path = os.path.join(root, name)
                logfile.write(gt_path+'\n')
                gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
                (gtdata[(gtdata[1].astype(float) < 20.) & (gtdata[3].astype(int) == 0)][2].value_counts()).to_string(
                    logfile)
                logfile.write('\n')
