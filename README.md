# nn.based.intersection.classficator

# TODO

- [ ] Create dataloader
- [ ] Create NN
- [ ] Try to classify using BEVs + LABELs

# Experiments were performed on the RTX Ivan PC.


# Data Folder structure.

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

# Video Generations

Given the folders' structure you can generate a mosaic of BEVs+Images using the script _generate.videos.bash_

# Info about bash scripts

The following scripts uses as input a txt file (foldes.txt) generated with 

```ls -d1 */ > folders.txt```

- generate.bev.and.pcds.sh
- generate.videos.bash