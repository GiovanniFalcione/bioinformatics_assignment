from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from scipy.stats import randint as sp_randint
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import pandas as pd

"""
Use Support Vector Machines to discriminate the subcellular location
of proteins given the features built from the data available (see
feature_eng.py).

author: Giovanni Falcione
bioinformatics assignment, Msc Machine Learning 16/17, UCL
"""

#load features and labels
features = np.load('feature_matrix.npy')
labels = np.load('labels.npy')

#get rid of unbalanced data
less_feat = np.vstack((features[:1300,:], features[3004:4303,:],features[4303:5603,:], features[-1300:,:]))
less_lab = np.hstack((labels[:1300], labels[3004:4303],labels[4303:5603], labels[-1300:]))

# split into training and test sets
n_train = less_feat.shape[0]
idx = range(n_train);np.random.shuffle(idx)
n_val = int(0.7*n_train)
idx_tr = idx[:n_val]
idx_te = idx[n_val:]

lab_tr = [less_lab[i] for i in idx_tr]
lab_te = [less_lab[i] for i in idx_te]

trainx = less_feat[idx_tr,:]
testx = less_feat[idx_te,:]
trainy = lab_tr
testy = lab_te

# print np.sum(np.asarray(less_lab) == 'cyto')
# print np.sum(np.asarray(less_lab) == 'mito')
# print np.sum(np.asarray(less_lab) == 'nucleus')
# print np.sum(np.asarray(less_lab) == 'secreted')

############### ML 
#grid search cross validation
n_iter = 15
param_grid = {"C": np.arange(47, 50 ,1),
              "gamma": np.logspace(-2,0,15)}

modd = SVC(probability= True, C= 121, gamma= 0.0021544)
modd.fit(trainx, trainy)

#make predictions on test set and print accuracy
my_pred1 = modd.predict(testx)
accuracy1 = np.sum(my_pred1 == testy)/float(n_train - n_val)
print 'accuracy: ' + str(accuracy1)


####------------ test on blind data
test_prots = np.load('blind_prots.npy')
test_ids = np.load('blind_ids.npy')
n_test = len(test_prots)
print n_test

#get prediction and relative probabilities
test_probs1 = modd.predict_proba(test_prots)
test_preds = modd.predict(test_prots)
test_preds_probs = np.max(test_probs1, axis=1)
test_preds2 = np.argmax(test_probs1, axis=1)

#put together IDs, predictions and probabilities in Pandas dataframe
df1 = pd.DataFrame({'prot ID': test_ids})
df2 = pd.DataFrame({'prot location': test_preds})
df3 = pd.DataFrame({'loc confidence': test_preds_probs})

results = pd.concat([df1, df2, df3], axis= 1)
print results

# build and display confusion matrix
y_confusion = np.zeros((4, 4))
idxi = 0
for i in ["cyto", "mito","nucleus", "secreted"]:
    idxj = 0
    for j in ["cyto", "mito","nucleus", "secreted"]:

        y_confusion[idxi, idxj] = np.sum((np.asarray(testy) == i) & (np.asarray(my_pred1) == j))
        idxj += 1
    idxi+=1
y_confusion = np.around(y_confusion / np.sum(y_confusion, axis=1, keepdims=1), 4)
print ' displaying confusion matrix'
print y_confusion

