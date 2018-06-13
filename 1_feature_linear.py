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
unk_test_df = pd.read_csv('../data/f_test_a_20180204.csv',encoding='gb2312')

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
del_feature.extend(['id'])

#删除缺失值大于一半的特征
feature = train_df.count()
drop_feature = feature[feature<500]
del_feature.extend(drop_feature.index)

train_df.drop(del_feature,axis=1, inplace=True)
unk_test_df.drop(del_feature,axis=1, inplace=True)

#统计类别型的特征,, 'BMI分类'算作连续特诊个类别特征
category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM家族史', 'ACEID'])

#得到目前还剩余的类别特征
category_feature = list(set(train_df.columns) & set(category_feature))
continuous_feature = list(set(train_df.columns) - set(category_feature))
#看一下连续特征的分布
#for col in continuous_feature:
#    col_temp = pd.DataFrame({col:unk_test_df[col]})
#    col_temp.hist()
    
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

#添加连续值与平均值的差值和差值的绝对值
add_col = []
for col in continuous_feature:
#    train_df[col+'_sub'] = train_df[col] - train_df[col].mean()
    train_df[col+'_abs'] = abs(train_df[col] - train_df[col].mean())
#    unk_test_df[col+'_sub'] = unk_test_df[col] - train_df[col].mean()
    unk_test_df[col+'_abs'] = abs(unk_test_df[col] - train_df[col].mean())
    add_col.extend([col+'_abs'])
continuous_feature.extend(add_col)

#对连续特征做归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(train_df.values)
train_df = pd.DataFrame(data=X, columns=train_df.columns)
X = min_max_scaler.transform(unk_test_df.values)
unk_test_df = pd.DataFrame(data=X, columns=unk_test_df.columns)

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

#对train_df剩下的连续特征做加减乘除和反除
#train_df.replace(to_replace=0, value=0.01, inplace=True)
#unk_test_df.replace(to_replace=0, value=0.01, inplace=True)
#imp_feature = ['年龄', 'TG', '孕前BMI','孕前体重','hsCRP','wbc','收缩压']
#combinations_feat = list(itertools.combinations(imp_feature,2))
#for add_col in combinations_feat:
#    col_str = add_col[0]+' + '+add_col[1]
#    train_df[col_str] = train_df[add_col[0]] + train_df[add_col[1]]
#    unk_test_df[col_str] = unk_test_df[add_col[0]] + unk_test_df[add_col[1]]
#    
#    col_str = add_col[0]+' - '+add_col[1]
#    train_df[col_str] = train_df[add_col[0]] - train_df[add_col[1]]
#    unk_test_df[col_str] = unk_test_df[add_col[0]] - unk_test_df[add_col[1]]
#    
#    col_str = add_col[0]+' * '+add_col[1]
#    train_df[col_str] = train_df[add_col[0]] * train_df[add_col[1]]
#    unk_test_df[col_str] = unk_test_df[add_col[0]] * unk_test_df[add_col[1]]
#    
#    col_str = add_col[0]+' / '+add_col[1]
#    train_df[col_str] = train_df[add_col[0]] / train_df[add_col[1]]
#    unk_test_df[col_str] = unk_test_df[add_col[0]] / unk_test_df[add_col[1]]
#    
#    col_str = add_col[1]+' / '+add_col[0]
#    train_df[col_str] = train_df[add_col[1]] / train_df[add_col[0]]
#    unk_test_df[col_str] = unk_test_df[add_col[1]] / unk_test_df[add_col[0]]
##数据做归一化
#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(train_df.values)
#train_df = pd.DataFrame(data=X, columns=train_df.columns)
#
#X = min_max_scaler.transform(unk_test_df.values)
#unk_test_df = pd.DataFrame(data=X, columns=unk_test_df.columns)

#计算相关度
def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)[0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df


#连续变量中选取单因子的相关度大于0.07的
C = 0.06
#C=0.15
print(C)
corr_df_single = cal_corrcoef(train_df, train_Y, train_df.columns)
corr01 = corr_df_single[corr_df_single.corr_value>=C]
corr01_col = corr01['col'].values.tolist()

train_linear = pd.DataFrame()
test_linear = pd.DataFrame()
train_linear = train_df[corr01_col]
test_linear = unk_test_df[corr01_col]

#离散变量中选取单因子的相关度大于0.07的
corr_one_hot = cal_corrcoef(train_one_hot_df, train_Y, train_one_hot_df.columns)
corr02 = corr_one_hot[corr_one_hot.corr_value>=C]
corr02_col = corr02['col'].values.tolist()

train_linear = pd.concat([train_linear, train_one_hot_df[corr02_col]], axis=1)
test_linear = pd.concat([test_linear, test_one_hot_df[corr02_col]], axis=1)

#转化一下，好读取单个相关度
corr_one_hot.set_index('col', inplace=True)
'''
将one-hot编码后的特征间做与、或、异或、同或处理
如果处理后特征的相关度都大于原特征的2倍则添加这个特征
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
    
#    if corr_and > (corr_0*2) and corr_and > (corr_1*2) and corr_and>C:
#        train_linear[col_and] = and_value
#        test_linear[col_and] = (test_one_hot_df[col[0]] & test_one_hot_df[col[1]]).values
#        i += 1
    if corr_or > (corr_0*2) and corr_or > (corr_1*2) and corr_or>C:
        train_linear[col_or] = or_value
        test_linear[col_or] = (test_one_hot_df[col[0]] | test_one_hot_df[col[1]]).values
        i += 1
#    if corr_xor > (corr_0*2) and corr_xor > (corr_1*2) and corr_xor>C:
#        train_linear[col_xor] = xor_value
#        test_linear[col_xor] = (test_one_hot_df[col[0]] ^ test_one_hot_df[col[1]]).values
#        i += 1
print('add ont hot feature:',i)

train_linear.to_csv('../data/linear_train.csv',encoding='gb2312')
test_linear.to_csv('../data/linear_test.csv',encoding='gb2312')
pickle.dump(train_Y, open('../data/train_Y', 'wb'))




















