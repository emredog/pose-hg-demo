#!/usr/bin/env python

import os
import h5py
import numpy as np

prediction_file = '/home/emredog/git/pose-hg-demo/preds/he_test_scaled.h5'

csv_folder = '/home/emredog/git/pose-hg-demo/preds/he_test_scaled_csv/'
# create folder 
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

im_w = 256
im_h = 256
orig_w = 640
orig_h = 480

scale_x = float(orig_w) / float(im_w)
scale_y = float(orig_h) / float(im_h)


f = h5py.File(prediction_file, 'r')



for k in f.keys():	
	pose = np.asarray(f[k])

	pose_orig = np.zeros(pose.shape)
	pose_orig[:,0] = pose[:,0] * scale_x
	pose_orig[:,1] = pose[:,1] * scale_y
	cfile = k[:-3] + 'csv'

	np.savetxt('%s%s' % (csv_folder, cfile), pose_orig, delimiter=',', fmt='%.3f')

	


