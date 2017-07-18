#!/usr/bin/env python

import os
import csv
import cPickle as pickle
from PIL import Image
import numpy as np

isTrain = True

#raw_data_root = '/home/edogan/data/AdaptiveViewpointSelection/HumanEva/'
raw_data_root = '/media/emredog/research-data/HumanEva-I/'
project_root = '/home/emredog/git/pose-hg-demo/'

im_size=256
orig_w = 640
orig_h = 480
half_im_size = im_size/2
ideal_height = im_size * (3.0/4.0)
nbParts = 20

#def prepare_data(data_root, raw_data_root, isTrain=True, im_size=224, nbParts = 20):

    
partNames = ['torsoProximal', 'torsoDistal', 'upperLArmProximal', 'upperLArmDistal', 'lowerLArmProximal', 'lowerLArmDistal', 'upperRArmProximal', 'upperRArmDistal', 'lowerRArmProximal', 'lowerRArmDistal', 'upperLLegProximal', 'upperLLegDistal', 'lowerLLegProximal', 'lowerLLegDistal', 'upperRLegProximal', 'upperRLegDistal', 'lowerRLegProximal', 'lowerRLegDistal', 'headProximal', 'headDistal']

GT_PATH = project_root + 'annot/he/'
if isTrain:
    IM_PATH = raw_data_root + 'set_TRAIN/'
else:
    IM_PATH = raw_data_root + 'set_VALIDATE/'

if isTrain:
    OUT_IM_PATH = project_root + 'images/he/train_' + str(im_size) + '_cropped/'
    CSV_OUT = project_root + 'images/he/train_' + str(im_size) + '_cropped.csv'
else:
    OUT_IM_PATH = project_root + 'images/he/test_' + str(im_size) + '_cropped/'
    CSV_OUT = project_root + 'images/he/test_' + str(im_size) + '_cropped.csv'


# delete output csv file if it exists
try:
    os.remove(CSV_OUT)
except OSError:
    pass


# create folder 
if not os.path.exists(OUT_IM_PATH):
    os.makedirs(OUT_IM_PATH)
           

if isTrain:
    print('Working on TRAIN set.')    
    outputFile = GT_PATH + 'HE_' + str(im_size) + '_train.pkl'
else:
    print('Working on TEST set.')        
    outputFile = GT_PATH + 'HE_' + str(im_size) + '_test.pkl'    


actAbrevs = {'ThrowCatch':'thr', 'Walking':'wal', 'Box':'box', 'Gestures':'ges', 'Jog':'jog'}

# csv reader:
if isTrain:
    csvfile=open(GT_PATH + 'Train_C1C2C3_HE20_GroundTruth.csv')
else:
    csvfile=open(GT_PATH + 'Validate_C1C2C3_HE20_GroundTruth.csv')
reader = csv.DictReader(csvfile)

fullDict = {}

counter = 0    

out_csvfile = open(CSV_OUT, 'wb')
csvwriter = csv.writer(out_csvfile, delimiter=',')
    
    
partCounter = 0
partList = list()
# for each entry:
for row in reader:                

    partCounter = partCounter+1

    # after 20 parts, write a new entry for a frame
    if partCounter == nbParts:            
        partList.append( [float(row['X']), float(row['Y'])] ) # append the last list of [X,Y]   

        if (row['Joint'] != partNames[-1] or len(partList)!=nbParts ): # check for errors
            print('Joint', row['Joint'])
            print('Err len', len(partList))
            raise ValueError('Part type mismatch!')

        partCounter = 0            

        subj = row['Subject']
        
        subj = 'S' + subj
        act = row['Action']            
        frame = row['Frame']
        view = row['View']
        # partX = float(row['X']) # fIXME : these two lines are unnecessary
        # partY = float(row['Y'])        


        try:
            im = Image.open('%s%s_%s_1_(C%s)/im%.4d.bmp' % (IM_PATH, subj, act, view, int(frame)))
        except Exception as e:
            print('Could not open image: %s%s_%s_1_(C%s)/im%.4d.bmp' % (IM_PATH, subj, act, view, int(frame)))
            print(e)


        # Calculate where to crop:
        pl = np.asarray(partList)
        minx = np.min(pl[:,0])
        miny = np.min(pl[:,1])
        maxx = np.max(pl[:,0])
        maxy = np.max(pl[:,1])

        bbox_width = maxx-minx
        bbox_height = maxy-miny

        # center of the person        
        cent_x = float(maxx-minx)/2.0 + minx
        cent_y = float(maxy-miny)/2.0 + miny

        # overall scale so that the person's height is 3/4th of the image:
        scale_y = ideal_height / bbox_height

        # resize the image:
        im = im.resize(( int(np.round(float(orig_w)*scale_y)), int(np.round(float(orig_h)*scale_y)) ))

        # re-position the center:
        cent_x = int(np.round(cent_x*scale_y))
        cent_y = int(np.round(cent_y*scale_y))

        # Resize images:
        x1 = cent_x - half_im_size
        y1 = cent_y - half_im_size
        x2 = cent_x + half_im_size
        y2 = cent_y + half_im_size

        im = im.crop((x1, y1, x2, y2)) # crop between two corners (x1,y1) and (x2,y2)

        # orig: s1_box_c1_im0386.jpg
        imfile = '%s_%s_c%s_im%.4d.jpg' % (subj.replace('S', 's'), actAbrevs[act], view, int(frame))
        imgPath = '%s%s' % (OUT_IM_PATH, imfile)




        csvwriter.writerow([imfile, str(scale_y), str(cent_x), str(cent_y)])

        # Save images:
        im.save(imgPath)
        
        frameDict = {'Subject':subj, 'Action':act, 'Frame':frame, 'PartXYs':partList, 'View':view, 'Path':imgPath}
        fullDict[counter] = frameDict # append to main dictionary
        counter = counter+1



        partList = list() # clears the list 
        if counter % 250 == 0 and counter > 0:                    
            print('Processed', counter)
    else:
        partList.append( [float(row['X']), float(row['Y'])] ) # append a list of [X,Y]

print('CSV is now parsed. Saving the output...')

f = open(outputFile, 'wb')
pickle.dump(fullDict, f, pickle.HIGHEST_PROTOCOL)
print('Done.')