# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:11:26 2016

@author: Administrator
"""
##mlp多层神经网络
from scipy import sparse
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import pandas as pd
import numpy as np
from scipy import sparse as ssp
import pylab as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.base import BaseEstimator
#from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_classif
from sklearn.linear_model import Ridge,LogisticRegression
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
#import xgboost as xgb

seed = 1024
max_index_label = 1021
dim = 128
lsi_dim = 300

path = "E:/workspace/kaggle/talking data/Talking data/"

print("read app events")
app_ev = pd.read_csv(path+"app_events/app_events.csv",error_bad_lines=False,dtype={'device_id':np.str})  #添加error_bad_lines 貌似有的行列数对不到
#remove duplicates(app_id)
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))  #得到： event_id 2     app_id:487766649788038994 app_id:7010312103145...
print("read events")
events = pd.read_csv(path+"events.csv/events.csv",error_bad_lines=False,dtype={'device_id':np.str})
events["app_id"] = events["event_id"].map(app_ev)
events = events.dropna()

del app_ev
events = events[["device_id","app_id"]]
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")   #得到两列包括device id 和 app_id，一个device id对应若干app id

events = pd.concat([pd.Series(row['device_id'],row['app_id'].split(' ')) for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id','device_id']  #从一对多拆分为一对一，一个device id对应一个app id

print("read phone brand")
pbd = pd.read_csv(path+"phone_brand_device_model.csv/phone_brand_device_model.csv",dtype={'device_id': np.str})
pbd.drop_duplicates('device_id',keep='first',inplace=True)

print("# Generate Train and Test")
train = pd.read_csv(path+"gender_age_train.csv/gender_age_train.csv",
                    dtype={'device_id': np.str})
train['gender'][train['gender']=='M']=1
train['gender'][train['gender']=='F']=0
Y_gender = train['gender']
Y_age = train['age']
Y_age = np.log(Y_age)
train.drop(["age", "gender"], axis=1, inplace=True)
test = pd.read_csv(path+"gender_age_test.csv/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["group"] = np.nan
split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)  #对训练label进行编码，得到int64类型的array
device_id = test["device_id"]

#concat
Df = pd.concat((train,test),axis=0,ignore_index=True) #train和test按行拼接
Df = pd.merge(Df,pbd,how="left",on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x :"phone_brand"+str(x))
Df["device_model"] = Df["device_model"].apply(lambda x :"device_model"+str(x))

#  Concat Feature
f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model
f3 = events[["device_id", "app_id"]]    # app_id
del Df

f1.columns.values[1] = "feature"  #得到两列数据 device id 和 feature。每列都包含train和test数据
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"
FLS = pd.concat((f1,f2,f3),axis=0,ignore_index=True)  #上述三个行合并，只有device和feature

# User-Item Feature
print("# User-Item Feature")
device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()
data = np.ones(len(FLS))   #生成与FLS相同行数1的array
dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])   #对device id进行编码得到int64的array
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix((data,(row,col)),shape=(len(device_ids),len(feature_cs)))   #构建稀疏矩阵，在行为“device——id”，列为“feature”上，设置元素1，
sparse_matrix = sparse_matrix[:,sparse_matrix.getnnz(0)>0]   #把那些整列都是0的列删除

#Data
train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row,:]
test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]

skf = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=seed)  #用于交叉验证
for ind_tr, ind_te in skf:
    X_train = train_sp[ind_tr]
    X_val = train_sp[ind_te]   #稀疏矩阵
    y_train = Y[ind_tr]   #Y:group
    y_val = Y[ind_te]
    y_train_gender = Y_gender[ind_tr]
    y_val_gender = Y_gender[ind_te]
    y_train_age = Y_age[ind_tr]
    y_val_age = Y_age[ind_te]

    break

#   Feature Sel
print("# Feature Selection")
selector = SelectPercentile(f_classif,percentile=23) #根据最高得分用于特征选择
selector.fit(X_train,y_train)
X_train = selector.transform(X_train).toarray()
X_val = selector.transform(X_val).toarray()
train_sp = selector.transform(train_sp)
test_sp = selector.transform(test_sp)   #删掉原程序中toarray。原程序报错
print("# Num of Features: ", X_train.shape[1])   #4822个特征 ，类似 one.hot编码
group_lb = LabelBinarizer()
labels = group_lb.fit_transform(Y)   #把Y的每个元素转化若干位的二进制向量
y_train = group_lb.transform(y_train)
y_val = group_lb.transform(y_val)

inputs = Input(shape=(X_train.shape[1],),dtype='float32')  #实例化一个Keras张量,行数对应特征个数
fc1 = Dense(512)(inputs)        #隐藏层维数512？
fc1 = SReLU()(fc1)              #隐藏层的激活函数，类似sigmod
dp1 = Dropout(0.5)(fc1)        #dropout防止过拟合

y_train = [y_train,y_train_gender,y_train_age]   #得到list类型数据,行向量与数值、数值相对应.  y_train[0][0,:]得到第一个list的第一行
y_val = [y_val,y_val_gender,y_val_age]
outputs_gender = Dense(1,activation='sigmoid',name='outputs_gender')(dp1)   #dense定义输出层维度为1
outputs_age = Dense(1,activation='linear',name='outputs_age')(dp1)
outputs = Dense(12,activation='softmax',name='outputs')(dp1)  #label总共有12种
inputs = [
            inputs,
        ]

outputs = [
            outputs,
            outputs_gender,
            outputs_age,
        ]
model = Model(input=inputs, output=outputs)
nadam = Nadam(lr=1e-4)
sgd = SGD(lr=0.005,decay=1e-6,momentum=0.9,nesterov=True)    #随机梯度下降（优化方法）
model.compile(optimizer=nadam,
              loss={'outputs': 'categorical_crossentropy', 'outputs_gender': 'binary_crossentropy','outputs_age':'mse'},
              loss_weights={'outputs': 1., 'outputs_gender': 1.,'outputs_age': 1.}
              )
model_name = 'mlp_%s.hdf5'%'sparse'
model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_outputs_loss', save_best_only=True)   #用于下文中的model.fix callback
plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)

nb_epoch = 20
batch_size = 12   #每次迭代的样本数目
load_model = True
#if load_model:
#    print('Load Model')
#    model.load_weights(path+model_name)  #ctrl+1
    
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], 
              validation_data=[X_val,y_val]
              )









