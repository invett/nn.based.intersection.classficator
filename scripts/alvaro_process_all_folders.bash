#!/bin/bash

#basic call within scripts folder
#CUDA_VISIBLE_DEVICES=0 python predict.py \
	#--data_dir /home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30/2011_09_30_drive_0018_sync \
	#--pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
	#--feature_type ganet \
	#--feature_pyramid \
	#--refinement_type hourglass \
	#--no_intermediate_supervision

# SALVAR COMO NPZ o PNG
# --save_type npz | png (png is the default value)




CUDA_VISIBLE_DEVICES=0

while read line
do 
	if [[ ${line:0:1} != "#" ]]
	then
		echo "Processing " $line
		python predict.py \
		--data_dir $line \
		--pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
		--feature_type ganet \
		--feature_pyramid \
		--refinement_type hourglass \
		--save_type npz \
		--no_intermediate_supervision 
	else
		echo "Skipping" $line
	fi
done < folders_alvaro.txt


# example of folders_alvaro.txt
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0018_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0020_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0027_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0028_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0033_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_09_30_drive_0034_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_10_03_drive_0027_sync
#/home/malvaro/Documentos/DualBiSeNet/data_raw/2011_10_03_drive_0034_sync
