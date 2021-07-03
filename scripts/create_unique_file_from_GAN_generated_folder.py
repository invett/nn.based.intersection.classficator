import random
import re
from os import listdir
from os.path import isfile, join

mypath = '/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/data'

# overwrite file if present(w) or (a) for append... better w
f = open("/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/prefix_all.txt", "w")

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#optional random.shuffle(onlyfiles)
for index,filename in enumerate(onlyfiles):
    print(filename,';', re.split('-|_', onlyfiles[index])[4], sep='')
    f.write('data/' + filename + ';' + re.split('-|_', onlyfiles[index])[4] + '\n')

f.close()

