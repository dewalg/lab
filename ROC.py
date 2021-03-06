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
	

#truncate the data
tData = [None]*len(testData)
for i in range(0,len(testData)):
	tData[i] = [testData[i][3]*0.22, testData[i][4]]

tData = np.array(tData)


test_dir = '111413_jurkats_tsa_1day_1uM_800_v1_C'

pos = sci.loadmat(gen_dir+pos_dirloc+'/'+pos_dirloc+'_Plotting_Dataset.mat')
neg = sci.loadmat(gen_dir+neg_dirloc+'/'+neg_dirloc+'_Plotting_Dataset.mat')
test = sci.loadmat(gen_dir+test_dir+'/'+test_dir+'_Plotting_Dataset.mat')

pos = pos['Data_set'].T
neg = neg['Data_set'].T
test = test['Data_set'].T


# pos = np.array([pos[3]*0.22, pos[4]])
# neg = np.array([neg[3]*0.22, neg[4]])
# test = np.array([test[3]*0.22, test[4]])
pos = np.array(pos)
neg = np.array(neg)
test = np.array(test)


true_labels = np.c_[np.zeros((1, len(pos.T))), np.ones((1, len(neg.T)))]

X = np.array(np.c_[pos, neg]);
y = np.array(true_labels);

s = np.array([X,y[0]])



X, y = shuffle(X.T, y[0], random_state=random_state)

# shuffle and split training and test sets

n_samples = len(X)
half = int(n_samples / 2)
X_train, X_test = X[0:half], X[half:-1]
y_train, y_test = y[0:half], y[half:-1]


# Run classifier
classifier = svm.SVC(kernel='linear', probability=True)
model = classifier.fit(X_train, y_train)
probas_ = model.predict_proba(X_test)

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print ("Area under the ROC curve : ", roc_auc)


# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")


'''TESTING THE MODEL'''
# counter = 0;
# for test in tData:
# 	test = np.array([test[0], test[1]])
# 	tlabels = model.predict(test.T)
	
# 	a = np.array([[0,0]])
# 	b = np.array([[0,0]])
# 	pl.figure()
# 	iter = 0
# 	for label in tlabels:
# 		label = int(label)
		
# 		if label == 1: 
# 			a = np.r_[a, [test.T[iter]]]
# 		elif label == 0:
# 			b = np.r_[b, [test.T[iter]]]
		
# 		iter += 1


# 	pgreen = len(a)/(len(b)+len(a))*100
# 	pblue = len(b)/(len(b)+len(a))*100
# 	pl.plot(a.T[0], a.T[1], 'go', label= pgreen)
# 	pl.plot(b.T[0], b.T[1], 'bo', label= pblue)
	
# 	pl.xlabel('size')
# 	pl.ylabel('deform')
# 	pl.title(dataName[counter])
# 	pl.legend(loc="lower right")
	
# 	print (dataName[counter])
# 	print ('% green: ', str(pgreen), ' AND % blue: ', str(pblue))
# 	print ('\n')
# 	counter += 1

pl.show()