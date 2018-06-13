# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:13:32 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime

linear_SVR = pd.read_csv('../data/linear_SVR20180305_111359.csv',header=None)
unlinear_XGB = pd.read_csv('../data/XGB_feat_select20180305_111408.csv',header=None)
unlinear_LGB = pd.read_csv('../data/lgb_feat_select20180305_111322.csv',header=None)

unk_test_preds = np.zeros((len(linear_SVR), 3))

unk_test_preds[:,0] = linear_SVR[0].values
unk_test_preds[:,1] = unlinear_XGB[0].values
unk_test_preds[:,2] = unlinear_LGB[0].values

unk_test_preds = [1 if sum(i)>=2 else 0 for i in unk_test_preds]

submission = pd.DataFrame({'pred':unk_test_preds})
submission.to_csv(r'../data/bagging_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,index=False, float_format='%.4f')