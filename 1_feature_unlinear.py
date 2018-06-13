# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 09:40:27 2018

@author: Liaowei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:35:17 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import itertools
from sklearn import preprocessing
import pickle

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

np.seterr(invalid='ignore')

train_df = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')
train_df_a = pd.read_csv('../data/f_test_a_20180204.csv', encoding='gb2312')
train_df_a.index += len(train_df)
train_df_Y = pd.read_csv('../data/f_answer_a_20180306.csv',header=None)
train_df_a['label'] = train_df_Y.values
train_df = pd.concat([train_df, train_df_a], axis=0)
print(train_df.shape)
unk_test_df = pd.read_csv('../data/f_test_b_20180305.csv',encoding='gb2312')
train_df.rename(columns={'年龄':'age', '孕次':'times_of_pregnancy', '产次':'parity', '身高':'tall', '孕前体重':'YQ_TZ', 'BMI分类':'BMI_cate','孕前BMI':'YQ_BMI','收缩压':'SSY','舒张压':'SZY', '分娩时':'FMS', '糖筛孕周':'TSYZ','DM家族史':'DM_JZS'}, inplace = True)

unk_test_df.rename(columns={'年龄':'age', '孕次':'times_of_pregnancy', '产次':'parity', '身高':'tall', '孕前体重':'YQ_TZ', 'BMI分类':'BMI_cate','孕前BMI':'YQ_BMI','收缩压':'SSY','舒张压':'SZY', '分娩时':'FMS', '糖筛孕周':'TSYZ','DM家族史':'DM_JZS'}, inplace = True)

#把label提出出来
train_Y = train_df['label']
train_df.drop(['label'], axis=1, inplace=True)

#删除相关度太高的特征
#corr = train_df[train_df.columns].corr()
#count = 0
del_feature = []
#for col in corr.columns:
#    corr_data = corr[col][:count]
#    corr_data = corr_data[corr_data>0.8]
#    del_feature.extend(corr_data.index.values)
#    count += 1
#del_feature = list(set(del_feature))
#del_feature.extend(['id'])

#删除缺失值大于一半的特征
feature = train_df.count()
#drop_feature = feature[feature<500]
drop_feature = feature[feature<540]
del_feature.extend(drop_feature.index)
del_feature = list(set(del_feature))

train_df.drop(del_feature,axis=1, inplace=True)
unk_test_df.drop(del_feature,axis=1, inplace=True)

#统计类别型的特征,, 'BMI分类'算作连续特诊个类别特征
category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM_JZS', 'ACEID'])

#得到目前还剩余的类别特征
category_feature = list(set(train_df.columns) & set(category_feature))
continuous_feature = list(set(train_df.columns) - set(category_feature))
#看一下连续特征的分布
#for col in continuous_feature:
#    col_temp = pd.DataFrame({col:unk_test_df[col]})
#    col_temp.hist()
    

#用众数填充缺失值


'''
hsCRP采用平均值填充
对数处理：BUN，TG，Lpa，ALT

CHO有一个大于40的异常值
HDLC有一个大于30的异常值
AroB有个大于200的异常值
ApoA1有10来个是大于20的异常值
'''
train_df['hsCRP'] = train_df['hsCRP'].fillna(train_df['hsCRP'].mean())
train_df['BUN'] = np.log1p(train_df['BUN'])
train_df['TG'] = np.log1p(train_df['TG'])
train_df['Lpa'] = np.log1p(train_df['Lpa'])
train_df['ALT'] = np.log1p(train_df['ALT'])

train_df.loc[train_df['CHO']>40, 'CHO'] = train_df.loc[train_df['CHO']<=40, 'CHO'].mean()
train_df.loc[train_df['HDLC']>30, 'HDLC'] = train_df.loc[train_df['HDLC']<=30, 'HDLC'].mean()
train_df.loc[train_df['ApoB']>200, 'ApoB'] = train_df.loc[train_df['ApoB']<=200, 'ApoB'].mean()
train_df.loc[train_df['ApoA1']>20, 'ApoA1'] = train_df.loc[train_df['ApoA1']<=20, 'ApoA1'].mean()

unk_test_df['hsCRP'] = unk_test_df['hsCRP'].fillna(train_df['hsCRP'].mean())
unk_test_df['BUN'] = np.log1p(unk_test_df['BUN'])
unk_test_df['TG'] = np.log1p(unk_test_df['TG'])
unk_test_df['Lpa'] = np.log1p(unk_test_df['Lpa'])
unk_test_df['ALT'] = np.log1p(unk_test_df['ALT'])

unk_test_df.loc[unk_test_df['CHO']>40, 'CHO'] = train_df.loc[train_df['CHO']<=40, 'CHO'].mean()
unk_test_df.loc[unk_test_df['HDLC']>30, 'HDLC'] = train_df.loc[train_df['HDLC']<=30, 'HDLC'].mean()
unk_test_df.loc[unk_test_df['ApoB']>200, 'ApoB'] = train_df.loc[train_df['ApoB']<=200, 'ApoB'].mean()
unk_test_df.loc[unk_test_df['ApoA1']>20, 'ApoA1'] = train_df.loc[train_df['ApoA1']<=20, 'ApoA1'].mean()

train_df.fillna(train_df.median(axis=0),inplace=True)
unk_test_df.fillna(train_df.median(axis=0),inplace=True)

#做类别转换
for col in category_feature:
    train_df[col] = train_df[col].astype(dtype='int64')
    unk_test_df[col] = unk_test_df[col].astype(dtype='int64')

'''
下面部分对train和test都要做的
'''

#添加连续值与平均值的差值和差值的绝对值
add_col = []
for col in continuous_feature:
    train_df[col+'_abs'] = abs(train_df[col] - train_df[col].mean())
    unk_test_df[col+'_abs'] = abs(unk_test_df[col] - train_df[col].mean())
    add_col.extend([col+'_sub', col+'_abs'])
continuous_feature.extend(add_col)


#计算出one_hot编码后的数据
train_one_hot_df = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(train_df[feature],prefix=feature)
    train_df.drop([feature], axis=1, inplace=True)
    train_one_hot_df = pd.concat([train_one_hot_df, feature_dummy], axis=1)

test_one_hot_df = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(unk_test_df[feature],prefix=feature)
    unk_test_df.drop([feature], axis=1, inplace=True)
    test_one_hot_df = pd.concat([test_one_hot_df, feature_dummy], axis=1)

#对连续特征做归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(train_df.values)
train_df = pd.DataFrame(data=X, columns=train_df.columns)
X = min_max_scaler.transform(unk_test_df.values)
unk_test_df = pd.DataFrame(data=X, columns=unk_test_df.columns)

#对train_df剩下的连续特征做加减乘除和反除
train_df.replace(to_replace=0, value=0.01, inplace=True)
unk_test_df.replace(to_replace=0, value=0.01, inplace=True)
temp_feature = ['VAR00007', 'TG', 'age', 'wbc', 'YQ_BMI', 'VAR00007_abs', 'CHO_abs','BUN_abs', 'BUN', 'TG_abs', 'tall', 'hsCRP', 'hsCRP_abs']
single_feature = train_df.columns
double_feature = []
combinations_feat = list(itertools.combinations(temp_feature,2))
for add_col in combinations_feat:
    col_str = add_col[0]+'+'+add_col[1]
    double_feature.extend([col_str])
    train_df[col_str] = train_df[add_col[0]] + train_df[add_col[1]]
    unk_test_df[col_str] = unk_test_df[add_col[0]] + unk_test_df[add_col[1]]
    
    col_str = add_col[0]+'-'+add_col[1]
    double_feature.extend([col_str])
    train_df[col_str] = train_df[add_col[0]] - train_df[add_col[1]]
    unk_test_df[col_str] = unk_test_df[add_col[0]] - unk_test_df[add_col[1]]
    
    col_str = add_col[0]+'*'+add_col[1]
    double_feature.extend([col_str])
    train_df[col_str] = train_df[add_col[0]] * train_df[add_col[1]]
    unk_test_df[col_str] = unk_test_df[add_col[0]] * unk_test_df[add_col[1]]
    
    col_str = add_col[0]+'/'+add_col[1]
    double_feature.extend([col_str])
    train_df[col_str] = train_df[add_col[0]] / train_df[add_col[1]]
    unk_test_df[col_str] = unk_test_df[add_col[0]] / unk_test_df[add_col[1]]
    
    col_str = add_col[1]+'/'+add_col[0]
    double_feature.extend([col_str])
    train_df[col_str] = train_df[add_col[1]] / train_df[add_col[0]]
    unk_test_df[col_str] = unk_test_df[add_col[1]] / unk_test_df[add_col[0]]
#数据做归一化
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(train_df.values)
train_df = pd.DataFrame(data=X, columns=train_df.columns)
X = min_max_scaler.transform(unk_test_df.values)
unk_test_df = pd.DataFrame(data=X, columns=unk_test_df.columns)


train_unlinear = pd.DataFrame()
test_unlinear = pd.DataFrame()

'''
将one-hot编码后的特征间做与、或、异或、同或处理
如果处理后特征的相关度都大于原特征的2倍则添加这个特征
'''
single_one_hot_feature = train_one_hot_df.columns
double_one_hot_feature = []
combinations_feat = list(itertools.combinations(single_one_hot_feature,2))
i = 0
for col in combinations_feat:
    col_and = col[1]+'&'+col[0]
    col_or = col[1]+'|'+col[0]
    col_xor = col[1]+'^'+col[0]
    
    double_one_hot_feature.extend([col_and])
    train_one_hot_df[col_and] = (train_one_hot_df[col[0]] & train_one_hot_df[col[1]]).values
    train_one_hot_df[col_or] = (train_one_hot_df[col[0]] | train_one_hot_df[col[1]]).values
    train_one_hot_df[col_xor] = (train_one_hot_df[col[0]] ^ train_one_hot_df[col[1]]).values
    
    test_one_hot_df[col_and] = (test_one_hot_df[col[0]] & test_one_hot_df[col[1]]).values
    test_one_hot_df[col_or] = (test_one_hot_df[col[0]] | test_one_hot_df[col[1]]).values
    test_one_hot_df[col_xor] = (test_one_hot_df[col[0]] ^ test_one_hot_df[col[1]]).values
    



#采用XGboost筛选特征 
import xgboost as xgb  
import operator  

def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1
#        print(i)
    outfile.close() 

params = {  
    'min_child_weight': 100,  
    'eta': 0.02,  
    'colsample_bytree': 0.7,  
    'max_depth': 12,  
    'subsample': 0.7,  
    'alpha': 1,  
    'gamma': 1,  
    'silent': 1,  
    'verbose_eval': True,  
    'seed': 12  
}  

#单个连续特征筛选
rounds = 20 
xgtrain = xgb.DMatrix(train_df[single_feature].values, label=train_Y)  
bst = xgb.train(params, xgtrain, num_boost_round=rounds)  

features = single_feature
ceate_feature_map(features) 

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 

df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()
#df.to_csv("../data/feat_importance.csv", index=False) 

plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.show()

#筛选出前13个特征
#imp_feature = df.feature[0:13].values
imp_feature = df.feature[0:10].values
train_unlinear = pd.concat([train_unlinear, train_df[imp_feature]], axis=1)
test_unlinear = pd.concat([test_unlinear, unk_test_df[imp_feature]], axis=1)


#筛选出双因子的重要特征
rounds = 20 
xgtrain = xgb.DMatrix(train_df[double_feature].values, label=train_Y)  
bst = xgb.train(params, xgtrain, num_boost_round=rounds)  

features = double_feature
ceate_feature_map(features) 

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 

df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()   

plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.show()

#imp_feature = df.feature[0:21].values
imp_feature = df.feature[0:40].values
train_unlinear = pd.concat([train_unlinear, train_df[imp_feature]], axis=1)
test_unlinear = pd.concat([test_unlinear, unk_test_df[imp_feature]], axis=1)


#单个onehot特征的筛选
rounds = 30 
xgtrain = xgb.DMatrix(train_one_hot_df[single_one_hot_feature].values, label=train_Y)  
bst = xgb.train(params, xgtrain, num_boost_round=rounds)  

features = single_one_hot_feature
ceate_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 

df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()   

plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.show()

imp_feature = df.feature[0:11].values
train_unlinear = pd.concat([train_unlinear, train_one_hot_df[imp_feature]], axis=1)
test_unlinear = pd.concat([test_unlinear, test_one_hot_df[imp_feature]], axis=1)

#双one-hot特征筛选
rounds = 30 
xgtrain = xgb.DMatrix(train_one_hot_df[double_one_hot_feature].values, label=train_Y)  
bst = xgb.train(params, xgtrain, num_boost_round=rounds)  

features = double_one_hot_feature
ceate_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 

df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()   

plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')
plt.show()

imp_feature = df.feature[0:24].values
#imp_feature = df.feature[0:20].values
train_unlinear = pd.concat([train_unlinear, train_one_hot_df[imp_feature]], axis=1)
test_unlinear = pd.concat([test_unlinear, test_one_hot_df[imp_feature]], axis=1)

train_unlinear.to_csv('../data/unlinear_train.csv',encoding='gb2312')
test_unlinear.to_csv('../data/unlinear_test.csv',encoding='gb2312')
pickle.dump(train_Y, open('../data/train_Y', 'wb'))