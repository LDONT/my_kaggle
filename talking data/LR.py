# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:36:15 2016

@author: Administrator
"""
import pandas as pd
import numpy as np
#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import scipy.io

datadir = 'E:/workspace/kaggle/talking data/Talking data'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),index_col='device_id')  #device id 中没有重复的len(gatrain.index.unique())，len(gatrain.index)
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),index_col='device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id').set_index('device_id')  #清洗数据
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     #parse_dates=['timestamp'], 
                     index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        #usecols=['event_id','app_id','is_active'],
                        error_bad_lines=False,             #添加error_bad_lines 貌似有的行列数对不到
                        dtype={'is_active':np.bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
gatrain['trainrow'] = np.arange(gatrain.shape[0]) #对行数编号0 1 2 。。。得到trainrow
gatest['testrow'] = np.arange(gatest.shape[0])
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']  #设置了index，直接赋值就可以
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))   #对brand特征构建稀疏矩阵
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))

m = phone.phone_brand.str.cat(phone.device_model)  #把brand和model字符串拼接起来
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))

#已安装app的特征
appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)   # events.index.names
                       .groupby(['device_id','app'])['app'].agg(['size'])   #默认设置了index为['device_id','app']                                           
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)  #how='left'只使用左边的key连接； left_index=True设置左边的index作为连接的key
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)   #按理说how=left才对，但是结果为空，最后设置为right。前两行merge left没错
                       .reset_index())   #重置index                                                
d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))                       
                       
#app label特征                     
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data:train shape{},test shape{}'.format(Xtr_label.shape,Xte_label.shape))                      

#concatenate all features
Xtrain = hstack((Xtr_brand,Xtr_model,Xtr_app,Xtr_label),format='csr')
Xtest = hstack((Xte_brand,Xte_model,Xte_app,Xte_label),format='csr')

scipy.io.mmwrite('Xtrain.mtx',Xtrain)
Xtrain = scipy.io.mmread("Xtrain.mtx")
scipy.io.mmwrite('Xtest.mtx',Xtest)
Xtest = scipy.io.mmread("Xtest.mtx")

#print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))
#
#cross validation
targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)  #对gatrain的所有行的group编码
nclasses = len(targetencoder.classes_)

def score(clf,random_state = 0):
    kf = StratifiedKFold(y,n_folds=5,shuffle=True,random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    for itrain,itest in kf:
        Xtr,Xte = Xtrain[itrain,:],Xtrain[itest,:]
        ytr,yte = y[itrain],y[itest]
        clf.fit(Xtr,ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        return log_loss(yte,pred[itest,:])
        print("{:.5f}".format(log_loss(yte, pred[itest,:])), end=' ')
    print('')
    return log_loss(y,pred)

#Cs = np.logspace(-3,0,4)
#res = []
#for C in Cs:
#    res.append(score(LogisticRegression(C = C)))
#plt.semilogx(Cs, res,'-o');      
score(LogisticRegression(C=0.02))     #c表示正则强度，c越小强度越大 ;njobs=-1表示用所有cpu训练数据，weight=balance表示平衡类别,加入weight之后并没有什么效果   
#score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))    
# 
#clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')      
#clf.fit(Xtrain,y)
#pred = pd.DataFrame(clf.predict_proba(Xtest),index = gatest.index,columns=targetencoder.classes_)
#pred.to_csv('LR_pred.csv',index=True)

#len(gatrain['gender'][gatrain['gender']=='M'])  47904
#len(gatrain['gender'][gatrain['gender']=='F'])   26741  数据分布不均匀
#import matplotlib.pyplot as plt
#plt.hist(gatrain['group'])

###新加特征
#active 状态 app的特征  加上此类特征c=0.02时最好 2.28060597无语
#actappevents = appevents[appevents['is_active']==1]
#actappencoder = LabelEncoder().fit(actappevents.app_id)
#actappevents['actapp'] = actappencoder.transform(actappevents.app_id)
#actnapps = len(actappencoder.classes_)
#actdeviceapps = (actappevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)   # events.index.names
#                       .groupby(['device_id','actapp'])['actapp'].agg(['size'])   #默认设置了index为['device_id','app']                                           
#                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)  #how='left'只使用左边的key连接； left_index=True设置左边的index作为连接的key
#                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)   #按理说how=left才对，但是结果为空，最后设置为right。前两行merge left没错
#                       .reset_index())   #重置index                                                
#d_act = actdeviceapps.dropna(subset=['trainrow'])
#Xtr_actapp = csr_matrix((np.ones(d_act.shape[0]), (d_act.trainrow, d_act.actapp)), 
#                      shape=(gatrain.shape[0],actnapps))
#d_act = actdeviceapps.dropna(subset=['testrow'])
#Xte_actapp = csr_matrix((np.ones(d_act.shape[0]), (d_act.testrow, d_act.actapp)), 
#                      shape=(gatest.shape[0],actnapps)) 
#                      
#Xtrain = hstack((Xtr_brand,Xtr_model,Xtr_app,Xtr_label,Xtr_actapp),format='csr')
#Xtest = hstack((Xte_brand,Xte_model,Xte_app,Xte_label,Xte_actapp),format='csr')
#print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

#events['time_hour']= events['timestamp'].str[8:13]
#events = events.drop(['timestamp','longitude', 'latitude'],axis=1)
#app_label_category19 = pd.read_csv(os.path.join(datadir,'app_label_category19.csv'),usecols=['app_id','label_id','general_groups_num'])
#app_label_category19 = app_label_category19.loc[app_label_category19.app_id.isin(appevents.app_id.unique())]
#app_label_category19['app'] = appencoder.transform(app_label_category19.app_id)
#actappevents = appevents[appevents['is_active']==1]
#actappevents = actappevents.drop(['is_installed','is_active'],axis=1)
#deviceappcate = (actappevents.merge(events[['device_id','time_hour']],how='left',left_on='event_id',right_index=True)
#                             .merge(gatrain[['trainrow']], how='left', left_on='device_id', right_index=True)
#                             .merge(gatest[['testrow']], how='left', left_on='device_id', right_index=True))
#timeencoder = LabelEncoder().fit(deviceappcate.time_hour)
#deviceappcate['time_cate'] = timeencoder.transform(deviceappcate.time_hour)
#ntimecate = len(timeencoder.classes_)                
#d = deviceappcate.dropna(subset=['trainrow'])
#Xtr_timecate = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.time_cate)), 
#                      shape=(gatrain.shape[0],ntimecate))
#d = deviceappcate.dropna(subset=['testrow'])
#Xte_timecate = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.time_cate)), 
#                      shape=(gatest.shape[0],ntimecate))                              
                             

#加cate特征，没卵用
#devicecate = (deviceapps[['device_id','app']]
#                .merge(app_label_category19[['app','general_groups_num']])
#                .groupby(['device_id','general_groups_num'])['app'].agg(['size'])
#                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
#                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
#                .reset_index())
#cateencoder = LabelEncoder().fit(devicecate.general_groups_num)
#devicecate['cate'] = cateencoder.transform(devicecate.general_groups_num)
#ncate = len(cateencoder.classes_)                
#d = devicecate.dropna(subset=['trainrow'])
#Xtr_cate = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.cate)), 
#                      shape=(gatrain.shape[0],ncate))
#d = devicecate.dropna(subset=['testrow'])
#Xte_cate = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.cate)), 
#                      shape=(gatest.shape[0],ncate))                    