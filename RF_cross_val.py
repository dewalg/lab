
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd
import numpy as np
import scipy.io as sci
import pylab as pl
import os

random_state = np.random.RandomState(0)

gen_dir = '/Users/dewal/Documents/lab/DC Data/Jurkat drug dosing/Jurkat TSA/TSA 3 day dosing 11_14/matlab data/'
pos_dirloc = '111613_jurkats_tsa_3day_control_800_v1_C'
neg_dirloc = '111613_jurkats_tsa_3day_2uM_800_v1_C'

allFileNames = os.listdir(gen_dir)
	
# load the positive and negative controls - create the training data
pos = sci.loadmat(gen_dir+pos_dirloc+'/'+pos_dirloc+'_Plotting_Dataset.mat')
neg = sci.loadmat(gen_dir+neg_dirloc+'/'+neg_dirloc+'_Plotting_Dataset.mat')

pos = np.array(pos['Data_set'])
neg = np.array(neg['Data_set'])

#create the label vector
label_pos = np.ones(len(pos))
label_neg = np.zeros(len(neg))

##### DATA VECTORS
y = np.concatenate([label_pos, label_neg])
x = np.concatenate([pos, neg])
n_samples, n_features = x.shape

#train the data
rf = RandomForestClassifier(n_estimators=100)

cv = StratifiedKFold(y, n_folds=10)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = rf.fit(x[train], y[train]).predict_proba(x[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
pl.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

pl.xlim([-0.01, 1.01])
pl.ylim([-0.01, 1.01])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()


















