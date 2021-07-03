import random
import re
from os import listdir
from os.path import isfile, join

mypath = '/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/data'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#optional
random.shuffle(onlyfiles)

items = len(onlyfiles)
trainset = onlyfiles[:int(items*0.7)]
validationset = onlyfiles[int(items*0.7):int(items*0.9)]
testset = onlyfiles[int(items*0.9):]

# overwrite file if present(w) or (a) for append... better w
f = open("/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/prefix_all.txt", "w")
for index,filename in enumerate(onlyfiles):
    print(filename,';', re.split('-|_', onlyfiles[index])[4], sep='')
    f.write('data/' + filename + ';' + re.split('-|_', onlyfiles[index])[4] + '\n')
f.close()

f = open("/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/prefix_train.txt", "w")
for index,filename in enumerate(trainset):
    f.write('data/' + filename + ';' + re.split('-|_', trainset[index])[4] + '\n')
f.close()

f = open("/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/prefix_val.txt", "w")
for index,filename in enumerate(validationset):
    f.write('data/' + filename + ';' + re.split('-|_', validationset[index])[4] + '\n')
f.close()

f = open("/media/14TBDISK/ballardini/GAN-generated_intersection_dataset-WARPED-conditional/prefix_test.txt", "w")
for index,filename in enumerate(testset):
    f.write('data/' + filename + ';' + re.split('-|_', testset[index])[4] + '\n')
f.close()