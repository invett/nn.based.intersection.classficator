import os
import shutil

path = '../../DualBiSeNet/kitti360-augusto/'

velodyne_path = '../../download_3d_velodyne/KITTI-360/data_3d_raw/'

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if 'left' in root:
            filepath = os.path.join(root, name)
            framename = name.split('_')[-1].replace('.png', '.bin')  # --frame name in velodyne
            seq_name = '_'.join(name.split('_')[:-1])
            aux_path, _ = os.path.split(root)
            new_file_path = os.path.join(aux_path, 'velodyne_points/data/')
            if not os.path.isdir(new_file_path):
                os.makedirs(new_file_path, 0o777)
            new_file_name = seq_name + '_' + framename
            destination = os.path.join(new_file_path, new_file_name)
            source = os.path.join(os.path.join(os.path.join(velodyne_path, seq_name), 'velodyne_points/data/'),
                                  framename)
            print('Coping file from {} to {}'.format(source, destination))
            dest = shutil.copy(source, destination)


