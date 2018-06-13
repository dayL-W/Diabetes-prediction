# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:09:17 2018

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
import xgboost as xgb

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
np.seterr(invalid='ignore')

train_df = pd.read_csv('../data/unlinear_train.csv',encoding='gb2312')
unk_test_df = pd.read_csv('../data/unlinear_test.csv',encoding='gb2312')
unk_test_Y = pd.read_csv('../data/f_answer_a_20180306.csv',header=None)
#train_df = pd.read_csv('../data/linear_train.csv',encoding='gb2312')
#unk_test_df = pd.read_csv('../data/linear_test.csv',encoding='gb2312')
train_Y = pickle.load(open('../data/train_Y', 'rb'))

kf = KFold(len(train_df), n_folds = 5, shuffle=True, random_state=520)


#params={'booster':'gbtree',
#    'objective': 'binary:logistic',
#    'eval_metric':'logloss',
#    'gamma':0,
#    'max_depth':4,
#    'lambda':1,
#    'eta':0.02,
#    'silent':0,
#    'alpha':7,
#    'seed':0,
#    'subsample':0.8,
#}
#n_round=900

#训练得分: 0.732181425486 linear
#线下得分: 0.665936473165
#线下得分: 0.731481481481
#params={'booster':'gbtree',
#    'objective': 'binary:logistic',
#    'eval_metric':'logloss',
#    'max_depth':6,
#    'min_child_weight': 50,
#    'lambda':6,
#    'eta':0.4,
#    'alpha':1,
#    'seed':0,
#    'subsample':1,
#}
#n_round=300

#训练得分: 0.736493936053
#线下得分: 0.683628318584
#线下得分: 0.742268041237
#params={'booster':'gbtree',
#    'objective': 'binary:logistic',
#    'eval_metric':'logloss',
#    'max_depth':4,
#    'min_child_weight': 50,
#    'lambda':2,
#    'eta':0.03,
#    'silent':0,
#    'alpha':2,
#    'subsample':0.95,
#}

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':5,
    'min_child_weight': 40,
    'lambda':16,
    'eta':0.2,
    'silent':0,
    'alpha':6,
    'subsample':1,
}
n_round=300
cv_preds = np.zeros(train_df.shape[0])
unk_test_preds = np.zeros((unk_test_df.shape[0], 5))
for i, (train_index, cv_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat = xgb.DMatrix(train_df.iloc[train_index].values, label=train_Y[train_index].values)
    cv_feat = xgb.DMatrix(train_df.iloc[cv_index].values)
    test_feat = xgb.DMatrix(unk_test_df.values)
    
    model = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round)
    cv_preds[cv_index] += model.predict(cv_feat)
    unk_test_preds[:,i] = model.predict(test_feat)
#看看训练结果
train_feat = xgb.DMatrix(train_df.values)
temp_train_preds = model.predict(train_feat)
temp_train_preds = [1 if i>=0.5 else 0 for i in temp_train_preds]
cv_preds = [1 if i>=0.5 else 0 for i in cv_preds]
#test_preds = np.mean(test_preds, axis=1)
#test_preds = [1 if i>=0.5 else 0 for i in test_preds]
print('训练得分:', f1_score(temp_train_preds, train_Y.values))
print('线下得分:', f1_score(cv_preds, train_Y.values))

#测试数据
unk_test_preds = np.mean(unk_test_preds, axis=1)
unk_test_preds = [1 if i>=0.5 else 0 for i in unk_test_preds]
#print('线下得分:', f1_score(unk_test_preds, unk_test_Y.values))
submission = pd.DataFrame({'pred':unk_test_preds})
submission.to_csv(r'../data/B_XGB_feat_select{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  header=None,index=False, float_format='%.4f')

