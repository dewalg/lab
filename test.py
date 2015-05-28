from __future__ import division
import os
import matplotlib.pyplot as matplot
import numpy as np
import scipy.io as sci

import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc


#np.set_printoptions(threshold='nan')
random_state = np.random.RandomState(0)

gen_dir = '/Users/dewal/Documents/lab/DC Data/Jurkat drug dosing/Jurkat TSA/TSA 3 day dosing 11_14/matlab data/'
pos_dirloc = '111613_jurkats_tsa_3day_control_800_v1_C'
neg_dirloc = '111613_jurkats_tsa_3day_2uM_800_v1_C'

allFileNames = os.listdir(gen_dir)

#find all files to be tested
testDataLoc = []
for iter in allFileNames:
	if iter[0] != '.':
		if iter != pos_dirloc and iter != neg_dirloc:
			testDataLoc.extend([iter])

#load all of those files
testData = [None]*len(testDataLoc)
dataName = [None]*len(testDataLoc)
counter = 0
for file in testDataLoc:
	x = sci.loadmat(gen_dir+file+'/'+file+'_Plotting_Dataset.mat')
	testData[counter] = x['Data_set'].T
	counter += 1

t = np.array(testData[1])
print(t.shape)