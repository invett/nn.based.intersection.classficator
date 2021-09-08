start from some folders that contains all the files in the SDCARD, let's say "raw_0001" for the first data-acquisition; 

**manually** create the output folders, lets say something like this:

.
├── gopro
│   ├── raw_0001
│   ├── raw_0002
│   ├── raw_0003
│   └── raw_0004
├── take_0001
├── take_0002
├── take_0003
└── take_0004

mkdir take_0001
mkdir take_0002
mkdir take_000...

raw_0001 -> take_0001
raw_0002 -> take_0002
raw_0003 -> take_0003
raw_0004 -> take_0004

inside each of the 'take_000x', do this to create sym-links to the MP4 files:

    for i in `ls ../gopro/raw_0001/DCIM/100GOPRO/*.MP4`; do ln -s $i; done;
    .....
    for i in `ls ../gopro/raw_000x/DCIM/100GOPRO/*.MP4`; do ln -s $i; done;

to extract frames form MP4.

    for i in $( ls *.MP4 ); do mkdir `basename $i .MP4`;  cd `basename $i .MP4`;  ffmpeg -i ../$i -vf scale=800:-1 %010d.png ;  cd ..; done;


GX010011.MP4
GX010012.MP4
GX020011.MP4
GX030011.MP4
GX040011.MP4
GX050011.MP4
GX060011.MP4
GX070011.MP4

GX010013.MP4
GX020013.MP4
GX030013.MP4
GX040013.MP4
GX050013.MP4
GX060013.MP4
GX070013.MP4

GX010014.MP4
GX010015.MP4
GX020015.MP4
GX030015.MP4
GX040015.MP4
GX050015.MP4
GX060015.MP4

GX010016.MP4
GX010017.MP4
GX010018.MP4
GX010019.MP4
GX020016.MP4
GX020018.MP4
GX020019.MP4
GX030016.MP4
GX040016.MP4

---

after that, I change my mind and added take_0001 to all the things inside the created folders with 

    for i in `ls */ -d`; do echo mv $i take_0001_$i; done;

mv GX010011/ take_0001_GX010011/
mv GX010012/ take_0001_GX010012/
mv GX020011/ take_0001_GX020011/
mv GX030011/ take_0001_GX030011/
mv GX040011/ take_0001_GX040011/
mv GX050011/ take_0001_GX050011/
mv GX060011/ take_0001_GX060011/
mv GX070011/ take_0001_GX070011/

this is because i wanted to move all the files inside the same folder, and to have some sort of separation while preserving the names at the same time, this is what came to my mind..


.(base) ballardini@ballardini-T14:/media/ballardini/500GBHECTOR/dataset/alcala-10.06.2021$ tree -d -L 1
.
├── take_0001_GX010011
├── take_0001_GX010012
├── take_0001_GX020011
├── take_0001_GX030011
├── take_0001_GX040011
├── take_0001_GX050011
├── take_0001_GX060011
├── take_0001_GX070011
├── take_0002_GX010013
├── take_0002_GX020013
├── take_0002_GX030013
├── take_0002_GX040013
├── take_0002_GX050013
├── take_0002_GX060013
├── take_0002_GX070013
├── take_0003_GX010014
├── take_0003_GX010015
├── take_0003_GX020015
├── take_0003_GX030015
├── take_0003_GX040015
├── take_0003_GX050015
├── take_0003_GX060015
├── take_0004_GX010016
├── take_0004_GX010017
├── take_0004_GX010018
├── take_0004_GX010019
├── take_0004_GX020016
├── take_0004_GX020018
├── take_0004_GX020019
├── take_0004_GX030016
└── take_0004_GX040016

do the selection with the labeling script (python) and then, since all the folders are super huge, do this to create the 'selected' folder

mkdir selected

script to do this, WARNING: EDIT 
    1. prefix 
    2. prefix_all_list.txt (if you want)
    3. delete the two ending echos
    
example:
while read file; do firstpart=$(echo $file | cut -d ';' -f 1); firstpartbasepath=$(echo $file | cut -d ';' -f 1); prefix='/media/ballardini/500GBHECTOR/dataset/alcala-10.06.2021'; firstpartbasepath=${firstpartbasepath#$prefix}; cutted=$(dirname $firstpartbasepath); echo mkdir -p selected$cutted; echo cp $firstpart selected$firstpartbasepath ; done < prefix_all_list.txt

example with no ending echos (**EXCECUTE THIS**):
while read file; do firstpart=$(echo $file | cut -d ';' -f 1); firstpartbasepath=$(echo $file | cut -d ';' -f 1); prefix='/media/ballardini/500GBHECTOR/dataset/alcala-10.06.2021'; firstpartbasepath=${firstpartbasepath#$prefix}; cutted=$(dirname $firstpartbasepath); mkdir -p selected$cutted; cp $firstpart selected$firstpartbasepath ; done < prefix_all_list.txt


**** FROM LOG.MD (inside the wiki folder in the code) ***


#### Checking the FIRST version of the dataset

cp prefix_all_list.txt copy.txt (for safety i wont use the original file)

for each of the previous lines with the txt files, to create videos, use this kind of "script":

(**EXCECUTE THESE LINES**)

cat copy.txt | grep ';0' > check_type_0.txt
cat copy.txt | grep ';1' > check_type_1.txt  
cat copy.txt | grep ';2' > check_type_2.txt  
cat copy.txt | grep ';3' > check_type_3.txt  
cat copy.txt | grep ';4' > check_type_4.txt  
cat copy.txt | grep ';5' > check_type_5.txt  
cat copy.txt | grep ';6' > check_type_6.txt

sed -i 's/;0//' check_type_0.txt
sed -i 's/;1//' check_type_1.txt
sed -i 's/;2//' check_type_2.txt
sed -i 's/;3//' check_type_3.txt
sed -i 's/;4//' check_type_4.txt
sed -i 's/;5//' check_type_5.txt
sed -i 's/;6//' check_type_6.txt

while read line; do echo file $line >> ffmpeg_check_type_0.txt; done < check_type_0.txt
while read line; do echo file $line >> ffmpeg_check_type_1.txt; done < check_type_1.txt
while read line; do echo file $line >> ffmpeg_check_type_2.txt; done < check_type_2.txt
while read line; do echo file $line >> ffmpeg_check_type_3.txt; done < check_type_3.txt
while read line; do echo file $line >> ffmpeg_check_type_4.txt; done < check_type_4.txt
while read line; do echo file $line >> ffmpeg_check_type_5.txt; done < check_type_5.txt
while read line; do echo file $line >> ffmpeg_check_type_6.txt; done < check_type_6.txt

(do not copy all the lines in bash, use a file instead...)
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_0.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_0.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_1.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_1.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_2.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_2.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_3.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_3.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_4.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_4.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_5.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_5.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_6.txt -c:v libx264 -profile:v high444 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_6.mp4

ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_0.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_0_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_1.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_1_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_2.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_2_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_3.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_3_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_4.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_4_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_5.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_5_h265.mp4
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_6.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p /tmp/alcala-26.01.2021_check_type_6_h265.mp4


tentative to write filenames (no success...)

ffmpeg -f image2 -i new.jpg -filter_complex drawtext="fontsize=50:fontcolor=white:fontfile=/usr/share/fonts/truetype/freefont/FreeSans.ttf:borderw=2:bordercolor=black:text='%{metadata\:source_basename\:NA}':x=100:y=100" out.avi
drawtext="fontsize=20:fontcolor=white:fontfile=FreeSans.ttf:text='%{metadata\:source_basename\:NA}':x=10:y=10"
ffmpeg -safe 0 -r 30 -f concat -i ffmpeg_check_type_0.txt -c:v libx265 -level:v 4.0 -pix_fmt yuv420p -filter_complex drawtext="fontsize=50:fontcolor=white:fontfile=/usr/share/fonts/truetype/freefont/FreeSans.ttf:borderw=2:bordercolor=black:text='%{metadata\:source_basename\:NA}':x=100:y=100" /tmp/alcala-26.01.2021_check_type_0_h265b.mp4
