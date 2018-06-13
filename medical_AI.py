# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:32:21 2018

@author: Liaowei
"""

'''
一、数据预处理
1、删除一些特征，id及缺失值太多的数据及重复相关的数据
2、对缺失值进行填充，然后对类别型数据进行编码
3、使用一些简单的分类器看看效果，
4、对结果进行融合

5、对于连续值选取3个相关度比较高的特征，离散数据和离散数据、连续数据怎么计算相关度，然后对连续值做标准化处理
6、对于离散的单因子特征，选取较高相关度的，对于双因子特征（取与、或、异或、）选取比单因子相关度高的特征
'''

import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import itertools

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

np.seterr(invalid='ignore')

train_df = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')
unk_test_df = pd.read_csv('../data/f_test_a_20180204.csv',encoding='gb2312')

train_Y = train_df['label']
train_df.drop(['label'], axis=1, inplace=True)


#删除相关度太高的特征
corr = train_df[train_df.columns].corr()
count = 0
del_feature = []
for col in corr.columns:
    corr_data = corr[col][:count]
    corr_data = corr_data[corr_data>0.8]
    del_feature.extend(corr_data.index.values)
    count += 1
del_feature = list(set(del_feature))
del_feature.extend(['id','身高'])
#print('del_feature:\n',del_feature)
#df_all.drop(del_feature,axis=1, inplace=True)


#删除缺失值大于一半的特征
feature = train_df.count()
drop_feature = feature[feature<500]
del_feature.extend(drop_feature.index)
del_feature = list(set(del_feature))

train_df.drop(del_feature,axis=1, inplace=True)
unk_test_df.drop(del_feature,axis=1, inplace=True)

#用众数填充缺失值
'''
hsCRP采用平均值填充
对数处理：BUN，TG，Lpa，ALT

CHO有一个大于40的异常值
HDLC有一个大于30的异常值
AroB有个大于200的异常值
ApoA1有10来个是大于20的异常值
'''
#train_df['hsCRP'] = train_df['hsCRP'].fillna(train_df['hsCRP'].mean())
#train_df['BUN'] = np.log1p(train_df['BUN'])
#train_df['TG'] = np.log1p(train_df['TG'])
#train_df['Lpa'] = np.log1p(train_df['Lpa'])
#train_df['ALT'] = np.log1p(train_df['ALT'])
#
#train_df.loc[train_df['CHO']>40, 'CHO'] = train_df.loc[train_df['CHO']<=40, 'CHO'].mean()
#train_df.loc[train_df['HDLC']>30, 'HDLC'] = train_df.loc[train_df['HDLC']<=30, 'HDLC'].mean()
#train_df.loc[train_df['ApoB']>200, 'ApoB'] = train_df.loc[train_df['ApoB']<=200, 'ApoB'].mean()
#train_df.loc[train_df['ApoA1']>20, 'ApoA1'] = train_df.loc[train_df['ApoA1']<=20, 'ApoA1'].mean()

train_df.fillna(train_df.median(axis=0),inplace=True)
unk_test_df.fillna(train_df.median(axis=0),inplace=True)


#统计类别型的特征,, 'BMI分类'课算作连续特诊个类别特征
category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM家族史', 'ACEID'])


category_feature = list(set(train_df.columns) & set(category_feature))
for col in category_feature:
    train_df[col] = train_df[col].astype(dtype='int64')
    unk_test_df[col] = unk_test_df[col].astype(dtype='int64')
continuous_feature = list(set(train_df.columns) - set(category_feature))
#
#for col in continuous_feature:
#    VAR007_temp = pd.DataFrame({col:unk_test_df[col]})
#    VAR007_temp.hist()
#计算相关度
def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)[0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df

'''
下面部分对train和test都要做的
'''
#计算出one_hot编码后的数据
train_one_hot_df = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(train_df[feature],prefix=feature)
    train_one_hot_df = pd.concat([train_one_hot_df, feature_dummy], axis=1)

test_one_hot_df = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(unk_test_df[feature],prefix=feature)
    test_one_hot_df = pd.concat([test_one_hot_df, feature_dummy], axis=1)

C = 0.06
#C=0.15
print(C)

#连续变量中选取单因子的相关度大于0.07的
corr_df_single = cal_corrcoef(train_df, train_Y, continuous_feature)
corr01 = corr_df_single[corr_df_single.corr_value>=C]
corr01_col = corr01['col'].values.tolist()
train_select = pd.DataFrame()
test_select = pd.DataFrame()
train_select = train_df[corr01_col]
test_select = unk_test_df[corr01_col]

#离散变量中选取单因子的相关度大于0.07的
corr_one_hot = cal_corrcoef(train_one_hot_df, train_Y, train_one_hot_df.columns)
corr02 = corr_one_hot[corr_one_hot.corr_value>=C]
corr02_col = corr02['col'].values.tolist()

train_select = pd.concat([train_select, train_one_hot_df[corr02_col]], axis=1)
test_select = pd.concat([test_select, test_one_hot_df[corr02_col]], axis=1)

#转化一下，好读取单个相关度
corr_one_hot.set_index('col', inplace=True)
'''
将one-hot编码后的特征间做与、或、异或、同或处理
如果处理后特征的相关度都大于原特征的1.5倍则添加这个特征
'''
combinations_feat = list(itertools.combinations(train_one_hot_df.columns,2))
i = 0
for col in combinations_feat:
    col_and = col[1]+' & '+col[0]
    col_or = col[1]+' | '+col[0]
    col_xor = col[1]+' ^ '+col[0]
    
    and_value = (train_one_hot_df[col[0]] & train_one_hot_df[col[1]]).values
    corr_and = abs(np.corrcoef(and_value,train_Y.values)[0,1])
    or_value = (train_one_hot_df[col[0]] | train_one_hot_df[col[1]]).values
    corr_or = abs(np.corrcoef(or_value,train_Y.values)[0,1])
    
    xor_value = (train_one_hot_df[col[0]] ^ train_one_hot_df[col[1]]).values
    corr_xor = abs(np.corrcoef(xor_value,train_Y.values)[0,1])
    
    corr_0 = corr_one_hot.loc[col[0]].values
    corr_1 = corr_one_hot.loc[col[1]].values
    
    if corr_and > (corr_0*2) and corr_and > (corr_1*2) and corr_and>C:
        train_select[col_and] = and_value
        test_select[col_and] = (test_one_hot_df[col[0]] & test_one_hot_df[col[1]]).values
        i += 1
#    if corr_or > (corr_0*2) and corr_or > (corr_1*2) and corr_or>C:
#        train_select[col_or] = or_value
#        test_select[col_or] = (test_one_hot_df[col[0]] | test_one_hot_df[col[1]]).values
#        i += 1
#    if corr_xor > (corr_0*2) and corr_xor > (corr_1*2) and corr_xor>C:
#        train_select[col_xor] = xor_value
#        test_select[col_xor] = (test_one_hot_df[col[0]] ^ test_one_hot_df[col[1]]).values
#        i += 1
print(i)

train_df = train_select
unk_test_df = test_select

#train_df, test_df, train_Y, test_Y = train_test_split(train_df, train_Y, test_size=0.1, random_state=0)
#
#train_df.reset_index(inplace=True, drop=True)
#test_df.reset_index(inplace=True, drop=True)
#train_Y.reset_index(drop=True, inplace=True)
#test_Y.reset_index(drop=True, inplace=True)

kf = KFold(len(train_df), n_folds = 5, shuffle=True, random_state=520)


#误差函数
def evalerror(pred, df):
    label = df.get_label().values.copy()
    pred = [1 if i>=0.5 else 0 for i in pred]
    score = f1_score(label,pred)
    #返回list类型，包含名称，结果，is_higher_better
    return ('F1',score,False)
#设置lightGBM的参数
#metric参数没有F1改使用哪个评价指标？
print('开始训练...')
params = {
    'num_leaves': 70,          #70 22
    'max_depth': 7,            #7 
    'min_data_in_leaf': 25,    #28
    'feature_fraction': 0.8,  #1
    'learning_rate': 0.48,     #0.36
    
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'verbose': -1,
    'metric': 'binary_logloss',
    'bagging_seed': 3,
}

cv_preds = np.zeros(train_df.shape[0])
unk_test_preds = np.zeros((unk_test_df.shape[0], 5))
#test_preds = np.zeros((test_df.shape[0], 5))
for i, (train_index, cv_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_df.iloc[train_index]
    cv_feat = train_df.iloc[cv_index]
    #试试对血糖值取对数
    lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
    lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
    gbm = lgb.train(params=params,                 #参数
                    train_set=lgb_train,             #要训练的数据
                    num_boost_round=6000,   #迭代次数
                    valid_sets=lgb_cv,  #训练时需要评估的列表
                    verbose_eval=False,       #
                    feval=evalerror,        #误差函数
                    
                    early_stopping_rounds=500)
    #评价特征的重要性
    feat_imp = pd.Series(gbm.feature_importance(), index=train_df.columns).sort_values(ascending=False)
    
    cv_preds[cv_index] += gbm.predict(cv_feat.values)
    unk_test_preds[:,i] = gbm.predict(unk_test_df.values)
#    test_preds[:,i] = gbm.predict(test_df.values)

#看看训练结果
temp_train_preds = gbm.predict(train_df.values)
temp_train_preds = [1 if i>=0.5 else 0 for i in temp_train_preds]
cv_preds = [1 if i>=0.5 else 0 for i in cv_preds]
#test_preds = np.mean(test_preds, axis=1)
#test_preds = [1 if i>=0.5 else 0 for i in test_preds]
print('训练得分:', f1_score(temp_train_preds, train_Y.values))
print('线下得分:', f1_score(cv_preds, train_Y.values))
#print('测试得分:', f1_score(test_preds, test_Y.values))
'''
目前参数训练得分0.855，线下得分0.665，存在很大的过拟合
'''

#测试数据
#unk_test_preds = np.mean(unk_test_preds, axis=1)
#unk_test_preds = [1 if i>=0.5 else 0 for i in unk_test_preds]
#submission = pd.DataFrame({'pred':unk_test_preds})
#submission.to_csv(r'../data/lgb_A_feat_select{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                  header=None,index=False, float_format='%.4f')


