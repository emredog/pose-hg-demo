#!/usr/bin/env python

import os
import h5py
import numpy as np

prediction_file = '/home/emredog/git/pose-hg-demo/preds/he_test_cropped.h5'
cropinfo_file = '/home/emredog/git/pose-hg-demo/images/he/test_256_cropped.csv'

csv_folder = '/home/emredog/git/pose-hg-demo/preds/he_test_croppedz_csv/'
# create folder 
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

im_size = 256
half_im_size = im_size/2
orig_w = 640
orig_h = 480

scale_x = float(orig_w) / float(im_w)
scale_y = float(orig_h) / float(im_h)


f = h5py.File(prediction_file, 'r')



for k in f.keys():	
	pose = np.asarray(f[k])


	# TODO: READ CROP INFO

	center = [0, 0]
	scale = 1.0

	tx = center[0] - half_im_size
	ty = center[1] - half_im_size


	pose_orig = np.zeros(pose.shape)
	pose_orig[:,0] = (pose[:,0]+tx) * 1.0/scale
	pose_orig[:,1] = (pose[:,1]+ty) * 1.0/scale
	cfile = k[:-3] + 'csv'

	np.savetxt('%s%s' % (csv_folder, cfile), pose_orig, delimiter=',', fmt='%.3f')

	


