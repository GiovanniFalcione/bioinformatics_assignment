from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from scipy.stats import randint as sp_randint
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import pandas as pd

#load features and labels
features = np.load('feature_matrix.npy')
labels = np.load('labels.npy')

print features.shape[1]

bubu_feat = np.vstack((features[:1300,:], features[3004:4303,:],features[4303:5603,:], features[-1300:,:]))
bubu_lab = np.hstack((labels[:1300], labels[3004:4303],labels[4303:5603], labels[-1300:]))

print np.unique(bubu_lab)
#n_train = features.shape[0]
n_train = bubu_feat.shape[0]
# split into training and test sets
idx = range(n_train);np.random.shuffle(idx)
n_val = int(0.7*n_train)
idx_tr = idx[:n_val]
idx_te = idx[n_val:]

lab_tr = [bubu_lab[i] for i in idx_tr]
lab_te = [bubu_lab[i] for i in idx_te]

trainx = bubu_feat[idx_tr,:]
testx = bubu_feat[idx_te,:]
trainy = lab_tr
testy = lab_te

print np.sum(np.asarray(bubu_lab) == 'cyto')
print np.sum(np.asarray(bubu_lab) == 'mito')
print np.sum(np.asarray(bubu_lab) == 'nucleus')
print np.sum(np.asarray(bubu_lab) == 'secreted')
#classify data using Random Forest

# # Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#
#
# # specify parameters and distributions to sample from
# param_dist = {"max_depth": sp_randint(1, 11),
#               #"max_features": sp_randint(1, 11),
#               "min_samples_split": sp_randint(2, 11),
#               "min_samples_leaf": sp_randint(1, 11)
#               }

# n_iter_search = 20
# mod_rf = GradientBoostingClassifier(n_estimators=20)
# rand_search = RandomizedSearchCV(mod_rf, param_distributions=param_dist, cv=4, n_iter= n_iter_search)
# rand_search.fit(trainx, trainy)
#
# report(rand_search.cv_results_)

#my_pred = rand_search.predict(testx)
#

#mod_svc1 = SVC(C= 95, gamma= 0.17, probability= True)
#
# # # print 'svm at work'
# for i in np.arange(50.1, 151, 10):
#     mod_svc1 = SVC(C= i, probability= True, kernel= 'poly', degree= 3)
#     print i
#     print np.mean(cross_val_score(mod_svc1, trainx, trainy , cv=4))

#after doing a grid search, the best parameters found on the
#first classifier are C = 50, gamma = 0.127 with acc: 0.6305


# mod_svc1 = SVC(probability= True, C= 50, gamma= 0.127)
# mod_svc1.fit(trainx, trainy)
# my_pred_prob_tr = mod_svc1.predict_proba(trainx)
# my_pred_prob_te = mod_svc1.predict_proba(testx)
#
# # #feed probabilities into second classifier
#
# mod_svc2 = SVC()
# grid_s = GridSearchCV(mod_svc2, param_grid= param_grid, cv = 4)
# grid_s.fit(my_pred_prob_tr, trainy)
#
#
# # #feed probabilities into second classifier that maps the
# # output probs of first SVM into final classification
#
# print grid_s.best_params_
#
# my_pred = grid_s.predict(my_pred_prob_te)

###############
n_iter = 15
param_grid = {"C": np.arange(47, 50 ,1),
              "gamma": np.logspace(-2,0,15)}

modd = SVC(probability= True, C= 121, gamma= 0.0021544)
modd.fit(trainx, trainy)
train_prob = modd.predict_proba(trainx)
test_prob = modd.predict_proba(testx)

my_pred1 = modd.predict(testx)
accuracy1 = np.sum(my_pred1 == testy)/float(n_train - n_val)
print 'accuracy 1: ' + str(accuracy1)

print testy[:15]
print my_pred1[:15]
# modd2 = SVC(probability= True, C = 47, gamma= 0.2683)
# # randd2 = RandomizedSearchCV(modd2, cv = 4, param_distributions= param_grid, verbose= True, n_iter= n_iter)
# modd2.fit(train_prob, trainy)
#
# #print randd2.best_params_
#
# my_pred2 = modd2.predict(scale(test_prob))
#
# accuracy2 = np.sum(my_pred2 == testy)/float(n_train - n_val)
# print 'accuracy 2: ' + str(accuracy2)


####---------------- test on blind data

test_prots = np.load('blind_prots.npy')
test_ids = np.load('blind_ids.npy')
n_test = len(test_prots)
print n_test

test_probs1 = modd.predict_proba(test_prots)

#test_probs2 = modd2.predict_proba(scale(test_probs1))

print test_probs1

test_preds = modd.predict(test_prots)
test_preds_probs = np.max(test_probs1, axis=1)
test_preds2 = np.argmax(test_probs1, axis=1)

df1 = pd.DataFrame({'prot ID': test_ids})
df2 = pd.DataFrame({'prot location': test_preds})
df3 = pd.DataFrame({'loc confidence': test_preds_probs})

results = pd.concat([df1, df2, df3], axis= 1)

print results

# build confusion matrix

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

