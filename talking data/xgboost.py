# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:50:52 2016

@author: Administrator
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.io
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#target = 'Disbursed'
#IDcol = 'ID'

#Xtrain = scipy.io.mmread("Xtrain.mtx")
#y = np.load("y.npy")
#Xtest = scipy.io.mmread("Xtest.mtx")

#def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
def modelfit(alg, y, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=5):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class']=12  #keypoint
#        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtrain = xgb.DMatrix(predictors,y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,  #是不是和watchlist功能差不多
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds,show_stdv=True,verbose_eval=1)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    #alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
    alg.fit(predictors,y,eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(predictors)  #返回最大概率的类别
    dtrain_predprob = alg.predict_proba(predictors)  #返回每个类别的概率
        
    #Print model report:
    print ("\nModel Report")
    print("{:.5f}".format(log_loss(y, dtrain_predprob)), end=' ')
    print ("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print ("Log Loss Score (Train): %f" % metrics.log_loss(y, dtrain_predprob))
    return (alg)
#    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
#    print (*"AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#    feat_imp.plot(kind='bar', title='Feature Importances')
#    plt.ylabel('Feature Importance Score')
 
#Step 1: Fix learning rate and number of estimators for tuning tree-based parameters   
#predictors = [x for x in train.columns if x not in [target, IDcol]]
predictors = Xtrain
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
# objective="gblinear",
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= "multi:softprob",
# n_classes=12,
 nthread=4,
# scale_pos_weight=1,
 seed=27)
modelfit(xgb1, y, predictors)
#预测
dtrain=xgb.DMatrix(predictors,y)
params=xgb1.get_xgb_params()
params['num_class']=12
model = xgb.train(dtrain=dtrain,params=params)
dtest=xgb.DMatrix(Xtest)
pred = pd.DataFrame(model.predict(dtest), index = gatest.index, columns=targetencoder.classes_)

#Step 2: Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':[7,9,10],  #10 12
 'min_child_weight':[5,7,9]  #9 15
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(predictors,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#Step 3: Tune gamma
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]  #0
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=12,
 min_child_weight=15, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch3.fit(predictors,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

#Step 4: Tune subsample and colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],  #0.6
 'colsample_bytree':[i/10.0 for i in range(6,10)]  #0.7
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=12,
 min_child_weight=15, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch4.fit(predictors,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

#Step 5: Tuning Regularization Parameters
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]  #1e-5
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=12,
 min_child_weight=15, gamma=0, subsample=0.6, colsample_bytree=0.7,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch6.fit(predictors,y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

#Step 6: Reducing Learning Rate
xgb4 = XGBClassifier( learning_rate =0.01, n_estimators=100, max_depth=12,
 min_child_weight=15, gamma=0, subsample=0.6, colsample_bytree=0.7,reg_alpha=0.00001,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1,seed=27) 

Xtrain = Xtrain.tocsr()
mask = np.random.choice([False, True], Xtrain.shape[0], p=[0.75, 0.25])
not_mask = ~mask
#kf = list(StratifiedKFold(y, n_folds=10, shuffle=True, random_state=4242))[0]
#Xtr, Xte = Xtrain[kf[0], :], Xtrain[kf[1], :]
#ytr, yte = y[kf[0]], y[kf[1]]
#print('Training set: ' + str(Xtr.shape))
#print('Validation set: ' + str(Xte.shape))
dtrain = xgb.DMatrix(Xtrain[not_mask], label=y[not_mask])
dtrain_watch = xgb.DMatrix(Xtrain[mask], label=y[mask])
dtest = xgb.DMatrix(Xtest)
evallist  = [(dtrain,'train'),(dtrain_watch, 'eval')]
dtrain = xgb.DMatrix(Xtrain,label=y)
params=xgb4.get_params()
params['num_class']=12
model = xgb.train(params=params, dtrain=dtrain, evals=evallist,early_stopping_rounds=4,verbose_eval=1,num_boost_round=100)
#model = xgb.train(params=params, dtrain=dtrain,verbose_eval=1,num_boost_round=100)
preds = pd.DataFrame(model.predict(dtest), index = gatest.index, columns=targetencoder.classes_)
preds.to_csv('LT_pred_xgboost2.csv',index=True)
#model = modelfit(xgb4,y,predictors)
#dtest=xgb.DMatrix(Xtest)
#pred1 = pd.DataFrame(model.predict_proba(dtest), index = gatest.index, columns=targetencoder.classes_)




