------------------------------------------------
01. CREATE THE PNG FROM THE MP4
------------------------------------------------

cd alcala-26.01.2021/
for i in $( ls *.MP4 ); do echo mkdir `basename $i .MP4`; echo cd `basename $i .MP4`; echo ffmpeg -i ../$i -vf scale=800:-1 %010d.png ;  echo cd ..; done;
for i in $( ls *.MP4 ); do mkdir `basename $i .MP4`;  cd `basename $i .MP4`;  ffmpeg -i ../$i -vf scale=800:-1 %010d.png ;  cd ..; done;


-----------------------------------------------
03. LABEL THE FRAMES
-----------------------------------------------

Use:        labelling-script.py

look at the script, you might change some lines...


-----------------------------------------------
02. CREATE THE LIST OF IMAGES
-----------------------------------------------

****THIS****
for each folder.... 

find . | grep png > all.txt
vim all.txt
while read file; do echo "$file"; done < all.txt

BETTER, but improvable ... this take a lot of time for the two `command`
while read file; do echo `basename $(pwd)`/`basename $file`; done < all.txt
***THIS****


****FOR NEW DATALOADER THAT USES****
    train_list.txt
    validation_list.txt
    test.txt

now, for each of the files if you want (or merge all three files using the part above within ***THIS***) we want to COPY only the "labelled" frames. An alternative is to recreate the png from the MP4 directly on the dgx... and who cares about the space

while read file; do mkdir -p selected/$file; cp $file selected/$file ; done < all.txt
    

-----------------------------------------------
04. CREATE SYMBOLIC LINKS
-----------------------------------------------

while read line; do folder=$(echo $line | cut -d '/' -f 1); filenamewithpath=$(echo $line | cut --d ';' -f 1); filename=$(echo $filenamewithpath | cut --d '/' -f 2); echo mkdir -p test/$folder; echo ln -s ../../$filenamewithpath test/$folder/$filename ; done < test_list.txt





-----------------------------------------------
04. CREATE THE WARPINGS
-----------------------------------------------

Use:        NOOOO   checkSequencesDataloader.py
            SIIII   generate-bev-dataset.py

check the part to use a single image to find the warping parameters.



    # USE THIS TO SELECT ONE SINGLE IMAGE AND VERIFY THE WARPING
    # for i in range(10):
    #     sample = dataloader.dataset.__getitem__(2030)
    #     data = sample['data']
    #     label = sample['label']
    #     # a = plt.figure()
    #     # plt.imshow(sample['data'] / 255.0)
    #     # send_telegram_picture(a, '')
    #     # plt.close('all')
    #
    # exit(1)
    
    54324
    
    **** WARNING!!! to execute the script from command line, use << python -m folder.script >> WITHOUT .py extension ****
                        see below for one example
    
    python -m scripts.generate-bev-dataset
    /home/ballardini/DualBiSeNet/alcala-12.02.2021/122302AA/0000008277.png
    Saving image in  /home/ballardini/DualBiSeNet/alcala-12.02.2021_augmented_warped
    Shutdown requested
    
                
                
            000 and 001 have been recorded on two cars, so , from the test_list.txt generated
            with all the pics from alvaro/augusto labelling, create two files to warp them with
            different values, but output the results in the same folder with the script
            
            000_test_list.txt
            001_test_list.txt            
            
            ./165810AA:     24199  001 C4
            ./164002AA:     27180  001 C4
            ./120445AA:     27420  000 ford
            ./122302AA:     26904  000 ford

            For images within 000, delete 164002AA and 165810AA
            :g/164002AA/d
            :g/165810AA/d

            For images within 001, delete 120445AA and 122302AA
            :g/120445AA/d
            :g/122302AA/d
    
    
    python -m scripts.generate-bev-dataset --augmentation 1 --workers=100 (on the dgx 100 is fine)
    
    now you'll have ONE or more <<output.txt>> and this file will act as the previous xxxx_list.txt like test_list.txt ; this file
    will hava a different structure for files
    
    122302AA/0000000791.png;0
    122302AA/0000000792.png;0
    122302AA/0000000793.png;0
    122302AA/0000000794.png;0
    
    will be now...
    
    122302AA/0000000798.001.png;0
    122302AA/0000000799.001.png;0
    122302AA/0000000797.001.png;0
    122302AA/0000000803.001.png;0
    
    create the symbolic links again for the warpings folder
    
    while read line; do folder=$(echo $line | cut -d '/' -f 1); filenamewithpath=$(echo $line | cut --d ';' -f 1); filename=$(echo $filenamewithpath | cut --d '/' -f 2); echo mkdir -p test/$folder; echo ln -s ../../$filenamewithpath test/$folder/$filename ; done < test_list.txt
    
    
    while read line; do folder=$(echo $line | cut -d '/' -f 1); filenamewithpath=$(echo $line | cut --d ';' -f 1); filename=$(echo $filenamewithpath | cut --d '/' -f 2); echo mkdir -p validation_test/$folder; echo ln -s ../../$filenamewithpath validation_test/$folder/$filename ; done < validation_test_list.txt
    
    cat 2nd.split.validation_list.txt.bugged  | cut -d ';' -f 1 > all_files_bugged.txt
    cat 2nd.split.validation_list.txt.bugged  | cut -d ';' -f 2 > all_annotations_bugged.txt
      
    120445AA/  122302AA
-----------------------------------------------
04. MIX LABEL RESULTS FROM ALVARO
-----------------------------------------------
cat 000.train_list.txt >> train_list.txt
cat 001.train_list.txt >> train_list.txt
cat 000.validation_list.txt >> validation_list.txt
cat 001.validation_list.txt >> validation_list.txt
cat 000.test_list.txt >> test_list.txt
cat 001.test_list.txt >> test_list.txt

or everything in test

cat 000.train_list.txt >> test_list.txt
cat 001.train_list.txt >> test_list.txt
cat 000.validation_list.txt >> test_list.txt
cat 001.validation_list.txt >> test_list.txt
cat 000.test_list.txt >> test_list.txt
cat 001.test_list.txt >> test_list.txt



wc -l 001.test_list.txt 001.train_list.txt 001.validation_list.txt 000.test_list.txt 000.train_list.txt 000.validation_list.txt
     853    001.test_list.txt
    3838    001.train_list.txt
    1225    001.validation_list.txt
     705    000.test_list.txt
    4290    000.train_list.txt
    1104    000.validation_list.txt
   12015    total







================== 
