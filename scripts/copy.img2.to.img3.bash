# THIS SCRIPT IS FOR COPYING THE 'SELECTED' FILES FROM THE 'DATA/RAW' FOLDER OF THE
# KITTI DATASET. ALVARO DID THIS ONLY FOR IMAGE_02; THIS IS THE SCRIPT THAT EXTRACTS
# THOSE IMAGES FROM THE FOLDER CONTAINING ALL THE IMAGE_03 TO 'OUR' IMAGE_O3, SO THAT
# IMAGE_02 AND IMAGE_03 HAS THE SAME FRAME-NUMBERS.

in_dir="/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_02"
data="/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_03/data/"
out_dir="/media/ballardini/4tb/ALVARO/Secuencias/2011_09_30_drive_0034_sync/image_03/"

for i in $( ls -v $in_dir/*.png )
do 
	image_02_filename=`cut -d '/' -f 9 <<< $i | tr -d _` 
	cp $data$image_02_filename $out_dir
	echo "Copying..." $i
done