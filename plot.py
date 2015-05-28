from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
import scipy.io as sci
import pylab as pl

random_state = np.random.RandomState(0)

gen_dir = '/Users/dewal/Documents/lab/DC Data/Jurkat drug dosing/Jurkat TSA/TSA 3 day dosing 11_14/matlab data/'
pos_dirloc = '111613_jurkats_tsa_3day_control_800_v1_C'
neg_dirloc = '111613_jurkats_tsa_3day_2uM_800_v1_C'


### DATA TO BE PLOTTED
dirloc = '111413_jurkats_tsa_1day_1uM_800_v1_C'
data = sci.loadmat(gen_dir+dirloc+'/'+dirloc+'_Plotting_Dataset.mat')

# load the positive and negative controls - create the training data
pos = sci.loadmat(gen_dir+pos_dirloc+'/'+pos_dirloc+'_Plotting_Dataset.mat')
neg = sci.loadmat(gen_dir+neg_dirloc+'/'+neg_dirloc+'_Plotting_Dataset.mat')

pos = np.array(pos['Data_set'])
neg = np.array(neg['Data_set'])
data = np.array(data['Data_set'])
#create the label vector
label_pos = np.ones(len(pos))
label_neg = np.zeros(len(neg))

##### DATA VECTORS
y = np.concatenate([label_pos, label_neg])
x = np.concatenate([pos, neg])
n_samples, n_features = x.shape

#train the data
x,y = shuffle(x,y,random_state=random_state)
rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(x, y)
class_label = rf.predict(data)

fig = pl.figure()
a = fig.add_subplot(111)
a.scatter(data[:,3]*0.22, data[:,4], c=class_label, lw=0)
pl.xlim([0, 30])
pl.ylim([0, 5])
pl.show()