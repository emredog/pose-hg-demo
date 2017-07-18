#!/usr/bin/env python

import os
import csv
import cPickle as pickle
from PIL import Image

isTrain = False

#raw_data_root = '/home/edogan/data/AdaptiveViewpointSelection/HumanEva/'
raw_data_root = '/media/emredog/research-data/HumanEva-I/'
project_root = '/home/emredog/git/pose-hg-demo/'

im_size=256
nbParts = 20

#def prepare_data(data_root, raw_data_root, isTrain=True, im_size=224, nbParts = 20):

    
partNames = ['torsoProximal', 'torsoDistal', 'upperLArmProximal', 'upperLArmDistal', 'lowerLArmProximal', 'lowerLArmDistal', 'upperRArmProximal', 'upperRArmDistal', 'lowerRArmProximal', 'lowerRArmDistal', 'upperLLegProximal', 'upperLLegDistal', 'lowerLLegProximal', 'lowerLLegDistal', 'upperRLegProximal', 'upperRLegDistal', 'lowerRLegProximal', 'lowerRLegDistal', 'headProximal', 'headDistal']

GT_PATH = project_root + 'annot/he/'
if isTrain:
    IM_PATH = raw_data_root + 'set_TRAIN/'
else:
    IM_PATH = raw_data_root + 'set_VALIDATE/'

if isTrain:
    OUT_IM_PATH = project_root + 'images/he/train_' + str(im_size) + '/'
else:
    OUT_IM_PATH = project_root + 'images/he/test_' + str(im_size) + '/'

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
        partX = float(row['X'])
        partY = float(row['Y'])


        try:
            im = Image.open('%s%s_%s_1_(C%s)/im%.4d.bmp' % (IM_PATH, subj, act, view, int(frame)))
        except Exception as e:
            print('Could not open image: %s%s_%s_1_(C%s)/im%.4d.bmp' % (IM_PATH, subj, act, view, int(frame)))
            print(e)

        # Resize images:
        im = im.resize((im_size,im_size))            

        # orig: s1_box_c1_im0386.jpg
        imgPath = '%s%s_%s_c%s_im%.4d.jpg' % (OUT_IM_PATH, subj.replace('S', 's'), 
                                                actAbrevs[act], view, int(frame)) 



        # Save images:
        im.save(imgPath)
        
        frameDict = {'Subject':subj, 'Action':act, 'Frame':frame, 'PartXYs':partList, 'View':view, 'Path':imgPath}
        fullDict[counter] = frameDict # append to main dictionary
        counter = counter+1



        partList = list() # clears the list 
        if counter % 1000 == 0 and counter > 0:
            print('Processed', counter)
    else:
        partList.append( [float(row['X']), float(row['Y'])] ) # append a list of [X,Y]

print('CSV is now parsed. Saving the output...')

f = open(outputFile, 'wb')
pickle.dump(fullDict, f, pickle.HIGHEST_PROTOCOL)
print('Done.')