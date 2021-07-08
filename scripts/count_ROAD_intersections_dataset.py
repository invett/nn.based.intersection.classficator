# this script is to count the intersections of

## ROAD: The ROad event Awareness Dataset for Autonomous Driving

## https://sites.google.com/view/roadchallangeiccv2021/call-for-papers?authuser=0
##  https://arxiv.org/pdf/2102.11585.pdf
##  https://github.com/gurkirt/road-dataset#frame-extraction

## labels here https://drive.google.com/drive/folders/1hCLlgRqsJBONHgwGPvVu8VWXxlyYKCq-

import json

# Opening JSON file
f = open('/tmp/road_trainval_v1.0.json')

# returns JSON object as
# a dictionary
data = json.load(f)

n=0
for i in data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames']:
    # print('\n')
    for j in data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames'][str(i)]:
        if data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames'][str(i)]['annotated'] == 1:
            for k in data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames'][str(i)]['annos']:
                for l in data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames'][str(i)]['annos'][str(k)]['loc_ids']:
                    # print(l)
                    if data['db']['2014-06-25-16-45-34_stereo_centre_02']['frames'][str(i)]['annos'][str(k)]['loc_ids'][0] == 9:
                        # print (i,' ',j,' ',k, ' ', l)
                        n=n+1
print(n)

# Closing file
f.close()
