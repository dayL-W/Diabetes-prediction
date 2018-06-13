# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 09:11:26 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import pickle
from sklearn.svm import SVC

import time
import datetime

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

np.seterr(invalid='ignore')

train_df = pd.read_csv('../data/linear_train.csv',encoding='gb2312')
unk_test_df = pd.read_csv('../data/linear_test.csv',encoding='gb2312')
unk_test_Y = pd.read_csv('../data/f_answer_a_20180306.csv',header=None)
train_Y = pickle.load(open('../data/train_Y', 'rb'))

kf = KFold(len(train_df), n_folds = 5, shuffle=True, random_state=0)


#误差函数
def evalerror(pred, df):
    label = df.get_label().values.copy()
    pred = [1 if i>=0.5 else 0 for i in pred]
    score = f1_score(label,pred)
    #返回list类型，包含名称，结果，is_higher_better
    return ('F1',score,False)

cv_preds = np.zeros(train_df.shape[0])
unk_test_preds = np.zeros((unk_test_df.shape[0], 5))
for i, (train_index, cv_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    
    
    
    train_feat = train_df.iloc[train_index]
    cv_feat = train_df.iloc[cv_index]
    
    clf = SVC(C=1, kernel='linear',max_iter=-1,tol=1e-4, gamma=1)
    clf.fit(X=train_feat.values, y=train_Y[train_index])
   
    cv_preds[cv_index] += clf.predict(cv_feat.values)
    unk_test_preds[:,i] = clf.predict(unk_test_df.values)

#from sklearn.model_selection import GridSearchCV
#grid = GridSearchCV(clf, param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
#grid.fit(X=train_df.values, y=train_Y.values)
#print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))

#看看训练结果
temp_train_preds = clf.predict(train_df.values)

print('训练得分:', f1_score(temp_train_preds, train_Y.values))
print('线下得分:', f1_score(cv_preds, train_Y.values))

#测试数据
unk_test_preds = [1 if sum(i)>=3 else 0 for i in unk_test_preds]
unk_test_preds = [1 if i>=0.5 else 0 for i in unk_test_preds]
print('线下得分:', f1_score(unk_test_preds, unk_test_Y.values))
#submission = pd.DataFrame({'pred':unk_test_preds})
#submission.to_csv(r'../data/linear_SVR{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                  header=None,index=False, float_format='%.4f')



