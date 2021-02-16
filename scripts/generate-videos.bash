#THIS SCRIPT CREATES THE VIDEOS USING THE SAME FOLDERS.TXT THAT SHOULD BE USED TO GENERATE THE BEVs AND PCDs

base=`pwd`
while read folder  
  do   
  if [[ ${folder:0:1} != "#" ]]
    then 
      cd $base
      echo "*************************************"
      echo "Processing" $folder
      echo "*************************************"
      cd `pwd`/$folder/bev 
      ffmpeg -nostdin -y -framerate 10 -i %*.png -vf scale=400:-1  -c:v libx264 $folder.bev.mp4
      cd $base
      cd `pwd`/$folder/image_02 
      ffmpeg -nostdin -y -framerate 10 -i %*.png -vf scale=400:-1  -c:v libx264 $folder.image_02.mp4
      cd $base
      ffmpeg -nostdin -y -i ./$folder/image_02/$folder.image_02.mp4 -i ./$folder/bev/$folder.bev.mp4 -filter_complex "nullsrc=size=400x522 [base]; [0:v] setpts=PTS-STARTPTS, scale=400x122 [upper]; [1:v] setpts=PTS-STARTPTS, scale=400x400 [lower]; [base][upper] overlay=shortest=1 [tmp1]; [tmp1][lower] overlay=shortest=1:y=122" -c:v libx264 $folder.mosaic.mp4
      cd $base
    else
      echo "Skipping " $folder
      echo "----------------"
  fi
  echo "FINE MP4" $folder
done < /media/ballardini/4tb/ALVARO/Secuencias/folders.txt
echo "END"
#ffmpeg -framerate 10 -i %*.png -vf scale=400:-1  -c:v libx264 test.mp4
#ffmpeg -i image_02/test.mp4 -i bev/test.mp4 -filter_complex "nullsrc=size=400x522 [base]; [0:v] setpts=PTS-STARTPTS, scale=400x122 [upper]; [1:v] setpts=PTS-STARTPTS, scale=400x400 [lower]; [base][upper] overlay=shortest=1 [tmp1]; [tmp1][lower] overlay=shortest=1:y=122" -c:v libx264 output.mp4