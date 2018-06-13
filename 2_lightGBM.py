# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:40:40 2018

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
import pickle

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

np.seterr(invalid='ignore')

#train_df = pd.read_csv('../data/linear_train.csv',encoding='gb2312')
#unk_test_df = pd.read_csv('../data/linear_test.csv',encoding='gb2312')
train_df = pd.read_csv('../data/unlinear_train.csv',encoding='gb2312')
unk_test_df = pd.read_csv('../data/unlinear_test.csv',encoding='gb2312')
unk_test_Y = pd.read_csv('../data/f_answer_a_20180306.csv',header=None)
train_Y = pickle.load(open('../data/train_Y', 'rb'))

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
#    'num_leaves': 30,          #70
    'max_depth': 3,            #7 
#    'min_data_in_leaf': 100,    #30
    'feature_fraction': 0.85  #1
#    'learning_rate': 0.48,     #0.48 0.36
#    
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'verbose': -1,
#    'metric': 'binary_logloss',
#    'bagging_seed': 3,
}

cv_preds = np.zeros(train_df.shape[0])
unk_test_preds = np.zeros((unk_test_df.shape[0], 5))
#test_preds = np.zeros((test_df.shape[0], 5))
for i, (train_index, cv_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = train_df.iloc[train_index]
    cv_feat = train_df.iloc[cv_index]

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
unk_test_preds = np.mean(unk_test_preds, axis=1)
unk_test_preds = [1 if i>=0.5 else 0 for i in unk_test_preds]
print('线下得分:', f1_score(unk_test_preds, unk_test_Y.values))
#submission = pd.DataFrame({'pred':unk_test_preds})
#submission.to_csv(r'../data/lgb_feat_select{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                  header=None,index=False, float_format='%.4f')
