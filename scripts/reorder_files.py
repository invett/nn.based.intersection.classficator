import os
import shutil

path = '../../DualBiSeNet/kitti360-augusto/'

velo_folder = 'velodyne_points/data/'

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        source = os.path.join(root, name)
        if 'pred' in root:
            seq_name = '_'.join(name.split('_')[:-2])  # --Sequence name
            framename = '_'.join(name.split('_')[-2:])  # --frame name
            new_folder_path = os.path.join(os.path.join(path, seq_name), 'pred')
            if not os.path.isdir(new_folder_path):
                print('Creating folder: {}'.format(new_folder_path))
                os.makedirs(new_folder_path, 0o777)
            destination = os.path.join(new_folder_path, framename)
            print('Coping file from {} to {}'.format(source, destination))
            shutil.move(source, destination)
        else:
            seq_name = '_'.join(name.split('_')[:-1])  # --Sequence name
            framename = name.split('_')[-1]  # --frame name
            if 'left' in root:
                new_folder_path = os.path.join(os.path.join(path, seq_name), 'image_02')
                if not os.path.isdir(new_folder_path):
                    print('Creating folder: {}'.format(new_folder_path))
                    os.makedirs(new_folder_path, 0o777)
                destination = os.path.join(new_folder_path, framename)
                print('Coping file from {} to {}'.format(source, destination))
                shutil.move(source, destination)
            if 'right' in root:
                new_folder_path = os.path.join(os.path.join(path, seq_name), 'image_03')
                if not os.path.isdir(new_folder_path):
                    print('Creating folder: {}'.format(new_folder_path))
                    os.makedirs(new_folder_path, 0o777)
                destination = os.path.join(new_folder_path, framename)
                print('Coping file from {} to {}'.format(source, destination))
                shutil.move(source, destination)
            if 'data' in root:
                new_folder_path = os.path.join(os.path.join(path, seq_name), velo_folder)
                if not os.path.isdir(new_folder_path):
                    print('Creating folder: {}'.format(new_folder_path))
                    os.makedirs(new_folder_path, 0o777)
                destination = os.path.join(new_folder_path, framename)
                print('Coping file from {} to {}'.format(source, destination))
                shutil.move(source, destination)

