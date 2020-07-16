# THIS SCRIPT SHOULD BE CALLED FROM THE FOLDER WHERE THE REPROJECT.PY IS LOCATED. ALL THE FOLDERS ARE THEN
# HARD CODED IN THIS SCRIPT.

# THIS SCRIPT CREATES THE PCDs AND THE BEVs USING 
# 	1. THE NPZ FILES GENERATED WITH aanet 
# 	2. THE ALVARO's MASKs

while read folder
do
  if [[ ${folder:0:1} != "#" ]]
  then 
    echo $folder
    for i in $( ls -v /media/ballardini/4tb/ALVARO/Secuencias/$folder/alvaromask/*.png )
      do alvaroFilename=`cut -d '/' -f 9 <<< $i | tr -d _` 
      alvaroFolder=`sed 's/aanet_//' <<< $i` 
      filenameKITTI=`sed "s|pred||" <<< $alvaroFilename`
      image_02=`sed "s|alvaromask|image_02|;s|pred||" <<< $i`
      image_03=`sed "s|alvaromask|image_03|;s|pred||" <<< $i`
      alvaroMaskFile=`sed 's/_pred/pred/' <<< $alvaroFolder`
      aanetFile=`sed "s|alvaromask|pred|;s|pred.png|_pred.png|;s|.png|.npz|" <<< $i`
      pcdName=`sed 's/pred.png//' <<< $alvaroFilename`
      pcdFolder=`sed "s|000.*||;s|aanet_||;s|alvaromask|pcd|;" <<< $i` 
      bevFolder=`sed "s|000.*||;s|aanet_||;s|alvaromask|bev|;" <<< $i` 
      echo python reproject.py $aanetFile $alvaroMaskFile out_aanet.ply $pcdFolder$pcdName.pcd $image_02 $image_03 $bevFolder$pcdName.png
      python reproject.py $aanetFile $alvaroMaskFile out_aanet.ply $pcdFolder$pcdName.pcd $image_02 $image_03 $bevFolder$pcdName.png
    done
    else
      echo "Skipping " $folder
      echo "----------------"
  fi
done < /media/ballardini/4tb/ALVARO/Secuencias/folders.txt