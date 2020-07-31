import os
import pandas as pd

path = '../../DualBiSeNet/data_raw/'

with open('labels.txt', 'w') as logfile:
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == 'frames_topology.txt':
                gt_path = os.path.join(root, name)
                logfile.write(gt_path + '\n')
                gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
                (gtdata[(gtdata[1].astype(float) < 20.) & (gtdata[3].astype(int) == 0)][2].value_counts()).to_string(
                    logfile)
                logfile.write('\n')
                
"""
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if '.png' in name and 'image_02' in root:
            frame, _ = os.path.splitext(name)
            file_path = os.path.join(root, name)
            maskname = name.replace('.png','pred.png')
            predname = name.replace('.png', '_pred.npz')
            mask_path = os.path.join(root, maskname)
            pred_path = os.path.join(root, predname)

            gtroot, _ = os.path.split(root)
            assert os.path.isfile(os.path.join(gtroot, 'frames_topology.txt')), 'wrong GT path '
            gt_path = os.path.join(gtroot, 'frames_topology.txt')
            gtdata = pd.read_csv(gt_path, sep=';', header=None, dtype=str)
            if ((gtdata[0] == frame) & (gtdata[3].astype(int) == 1)).any():
                os.remove(file_path)
                os.remove(file_path.replace('image_02', 'image_03'))
                os.remove(mask_path.replace('image_02', 'alvaromask'))
                os.remove(file_path.replace('image_02', 'altdiff'))
                os.remove(file_path.replace('image_02', 'bev'))
                os.remove(pred_path.replace('image_02', 'pred'))
                
"""

