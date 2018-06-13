# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:41:26 2018

@author: Liaowei
"""
'''
数据查看
1、查看每个特征的缺失数量，数据缺失太多的直接丢弃
2、查看每个特征的众数，用众数来填补缺失值
3、筛选出类别型数据和连续型数据
4、查看每个特征的相关系数，强相关的特征去除
5、查看连续值的分布情况，对连续值做处理
6、
'''
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

train_df = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')

feature = train_df.count()
drop_feature = feature[feature<500]
print('count:\n',drop_feature)
print('median:\n',train_df.median())

category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM家族史', 'ACEID'])

corr = train_df[train_df.columns].corr()
plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()

'''
孕前体重和孕前BMI，BMI分类及分娩时有很大的相关性
'''
count = 0
del_feature = []
for col in corr.columns:
    corr_data = corr[col][:count]
    corr_data = corr_data[corr_data>0.8]
    del_feature.extend(corr_data.index.values)
    count += 1
del_feature = list(set(del_feature))    
'''
VAR00007基本上是正态分布
'''
VAR007_temp = pd.DataFrame({'VAR':train_df['VAR00007'],'log(VAR+1)':np.log1p(train_df['VAR00007'])})
VAR007_temp.hist()
#train_df.groupby('label').mean().plot(y='VAR00007',marker='o')

VAR007_temp = pd.DataFrame({'VAR':train_df['VAR00007']})
VAR007_temp.hist()















