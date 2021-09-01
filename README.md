## Urban Intersection Classification: A Comparative Analysis

### License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-sa].
[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

This code is a WIP (Work In Progress), use at your own risk. This version only works on GPUs (no CPU version available).

Tested on:
* Kubuntu 20.04
* python 3.7
* cuda 11
* pytorch 1.8

# Info: 
## Data Folder structure

- The camera images are taken from the original [KITTI RAW dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php), synchronized version. 
- Crossing-frames and ground truth label are those available in [Intersection Ground Truth](https://ira.disco.unimib.it/research/robotic-perception-research/road-layout-estimation/an-online-probabilistic-road-intersection-detector/intersection-ground-truth/)
- files in *alvaromask* are created with **DualBiSeNet** (this correspond to the IV2020 Alvaro Submission)
- files in *pred* are created with **aanet** ; this are numpy-compressed images created with this version of [aanet](https://github.com/invett/aanet)
- files in *bev* and *pcd* are created with the **reproject.py** Python script; the PCD creation needs the PCL-Library installed as it uses *pcl_ply2pcd*; you don't need to call directly this script, but instead use the *generate.bev.and.pcds.sh* bash script in /scripts 

```.
├── 2011_09_30_drive_0018_sync
│   ├── alvaromask
│   ├── bev
│   ├── image_02
│   ├── image_03
│   ├── pcd
│   └── pred
├── 2011_09_30_drive_0020_sync
│   ├── alvaromask
│   ├── bev
│   ├── image_02
│   ├── image_03
│   ├── pcd
│   └── pred
.
.
```

## Usage

### reproject.py and generate.bev.and.pcds.sh

reproject takes as input the following list of arguments

- aanet output (disparity) as numpy npz, not the image. various test were performed as the original aaanet code saves in png with ```skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))``` and the results of the projection were not good.
- alvaro mask
- the intermediate output for pcl generation. will be auto deleted 
- PCD OUTPUT FILE NAME
- image_02 (from kitti)
- image_03 (from kitti)
- BEV OUTPUT FILE NAME


Example for Pycharm debug:
```
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/pred/0000001018_pred.npz
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/alvaromask/0000001018pred.png
out_aanet.ply 
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/pcd/0000001018.pcd 
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_02/0000001018.png 
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_03/0000001018.png 
/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/bev/0000001018.png
```

example output during execution:

```
python reproject.py /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/pred/0000001017_pred.npz /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/alvaromask/0000001017pred.png out_aanet.ply /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/pcd/0000001017.pcd /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_02/0000001017.png /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_03/0000001017.png /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/bev/0000001017.png
out_aanet.ply saved
Convert a PLY file to PCD format. For more information, use: pcl_ply2pcd -h
PCD output format: binary
> Loading out_aanet.ply [done, 110 ms : 45604 points]
Available dimensions: x y z rgb
> Saving /media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/pcd/0000001017.pcd [done, 2 ms : 45604 points]
```

## Video Generation

Given the folders' structure you can generate a mosaic of BEVs+Images using the script _generate.videos.bash_

## Info about BASH scripts

The following scripts uses as input a txt file (foldes.txt) generated with 

```ls -d1 */ > folders.txt```

- generate.bev.and.pcds.sh
- generate.videos.bash
- copy.img2.to.img3.bash: this script copies the "selected" images from original KITTI/DATA folder (example image_03/data) to somewhere else using the filenames contained in some other folder, example image_02
