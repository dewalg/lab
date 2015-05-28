
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
import numpy as np
import scipy.io as sci
import os
import pylab as pl


random_state = np.random.RandomState(0)

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_



feature_ranking = True
fit_data = True
fit_data_rfe = False
fit_data_rfe_cv = False
feat_roc = False
n_trees = 100
n_features_select = 10

##################
# GET THE DATA AND CLEANSE IT

gen_dir = '/Users/dewal/Documents/lab/DC Data/Jurkat drug dosing/Jurkat TSA/TSA 3 day dosing 11_14/matlab data/'
pos_dirloc = '111613_jurkats_tsa_3day_control_800_v1_C'
neg_dirloc = '111613_jurkats_tsa_3day_2uM_800_v1_C'

allFileNames = os.listdir(gen_dir)

#find all files to be tested
testDataLoc = []
for iter in allFileNames:
	if iter[0] != '.':    								#make sure its not a hidden directory
		if iter != pos_dirloc and iter != neg_dirloc:	#any other directory other than pre-set controls
			testDataLoc.extend([iter])

#load all of those files
testData = [None]*len(testDataLoc)
dataName = [None]*len(testDataLoc)
counter = 0
for file in testDataLoc:
	x = sci.loadmat(gen_dir+file+'/'+file+'_Plotting_Dataset.mat')
	testData[counter] = np.array(x['Data_set'])
	dataName[counter] = str(file)
	counter += 1
	
# load the positive and negative controls - create the training data
pos = sci.loadmat(gen_dir+pos_dirloc+'/'+pos_dirloc+'_Plotting_Dataset.mat')
neg = sci.loadmat(gen_dir+neg_dirloc+'/'+neg_dirloc+'_Plotting_Dataset.mat')

pos = np.array(pos['Data_set'])
neg = np.array(neg['Data_set'])

##################
# CREATE THE DATA VECTORS

label_pos = np.ones(len(pos))
label_neg = np.zeros(len(neg))
y = np.concatenate([label_pos, label_neg])
x = np.concatenate([pos, neg])
n_samples, n_features = x.shape

##################
# TRAIN THE CLASSIFIER(S)
if (fit_data):
	x,y = shuffle(x,y,random_state=random_state)
	rf = RandomForestClassifier(n_estimators=n_trees)
	rf = rf.fit(x, y)

## RFE (no CV)
if (fit_data_rfe):
	rf_feat_sel = RandomForestClassifierWithCoef(n_estimators=n_trees)
	rfe = RFE(rf_feat_sel, n_features_select, verbose=0)
	rfe = rfe.fit(x,y)

## RFE with CV
if (fit_data_rfe_cv):
	rf_feat_sel = RandomForestClassifierWithCoef(n_estimators=n_trees)
	rfe_cv = RFECV(rf_feat_sel, scoring='accuracy')
	rfe_cv = rfe_cv.fit(x,y)

##################
# PRINT THE FEAUTURE SELECTION/RANKING

if (feature_ranking):
	importances = rf.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rf.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]

	print("Feature ranking:")

	for f in range(n_features):
	    print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))

##################
# print the classification (a ratio) for 
# the samples in our original folder

if (fit_data):
	for i in range(0, len(testData)):
		pred = rf.predict(testData[i])
		live = sum(pred)
		ratio = live/len(testData[i])
		print("%%live: ",ratio, "| name: ", dataName[i])

##################
# RANDOM FOREST WITH RFE (NO CV)

if (fit_data_rfe):
	for i in range(0, len(testData)):
		pred = rfe.predict(testData[i])
		live = sum(pred)
		ratio = live/len(testData[i])
		print("%%live: ",ratio, "| name: ", dataName[i])

	importances = rfe.ranking_
	indices = np.argsort(importances)
	print("Feature ranking:")

	for f in range(n_features):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

##################
# RANDOM FOREST WITH RFE WITH CV

if (fit_data_rfe_cv):
	for i in range(0, len(testData)):
		pred = rfe_cv.predict(testData[i])
		live = sum(pred)
		ratio = live/len(testData[i])
		print("%%live: ",ratio, "| name: ", dataName[i])

	importances = rfe_cv.ranking_
	indices = np.argsort(importances)
	print("Feature ranking:")

	for f in range(n_features):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

if(feat_roc):
	half = int(n_samples / 2)
	x,y = shuffle(x,y,random_state=random_state)
	X_train, X_test = x[0:half], x[half:-1]
	y_train, y_test = y[0:half], y[half:-1]

	rf_feat_sel = RandomForestClassifierWithCoef(n_estimators=n_trees)

	for i in range(n_features):
		print(i)
		rfe = RFE(rf_feat_sel, i+1)
		rfe = rfe.fit(X_train,y_train)
		probas_ = rfe.predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
		roc_auc = auc(fpr, tpr)
		if (i==0 or i==18):
			print ("auc: ", roc_auc)
		pl.plot(fpr, tpr, lw=1)



	pl.xlim([-0.001, 1.001])
	pl.ylim([-0.001, 1.001])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic example')
	pl.show()








