# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:37:00 2018

@author: Liaowei
"""



'''
一、数据预处理
1、删除一些特征，id及缺失值太多的数据及重复相关的数据
2、对缺失值进行填充，然后对类别型数据进行编码
3、使用一些简单的分类器看看效果，
4、对结果进行融合
5、挑选出一些好的特征，删除一些不好的，同时自己构造一些特征
'''

import pandas as pd
import numpy as np
import math
import h5py
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb


train_df = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')
test_df = pd.read_csv('../data/f_test_a_20180204.csv',encoding='gb2312')
test_df.index += len(train_df)

train_Y = train_df['label']
train_df.drop(['label'], axis=1, inplace=True)

df_all = pd.concat([train_df, test_df], axis=0)
#删除ID
df_all.drop(['id'],axis=1, inplace=True)

#删除相关度太高的特征
corr = df_all[df_all.columns].corr()
count = 0
del_feature = []
for col in corr.columns:
    corr_data = corr[col][:count]
    corr_data = corr_data[corr_data>0.8]
    del_feature.extend(corr_data.index.values)
    count += 1
del_feature = list(set(del_feature))
print('del_feature:\n',del_feature)
df_all.drop(del_feature,axis=1, inplace=True)


#删除缺失值大于一半的特征
feature = df_all.count()
drop_feature = feature[feature<600]
print('drop_feature:\n',drop_feature)
df_all.drop(drop_feature.index,axis=1, inplace=True)

#用众数填充缺失值
#print('median:\n',df_all.median())
df_all.fillna(df_all.median(axis=0),inplace=True)

#统计类别型的特征
category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM家族史', 'ACEID'])

a = set(category_feature)
b  =set(drop_feature.index)
c =set(del_feature)

a = a - ((a&b) | (a&c))
category_feature = list(a)
#对类别型数据进行one-hot编码
for feature in category_feature:
    feature_dummy = pd.get_dummies(df_all[feature],prefix=feature)
    df_all.drop([feature], axis=1, inplace=True)
    df_all = pd.concat([df_all, feature_dummy], axis=1)

#误差函数
def evalerror(pred, df):
    label = df.get_label().values.copy()
    pred = [1 if i>=0.5 else 0 for i in pred]
    score = f1_score(label,pred)
    #返回list类型，包含名称，结果，is_higher_better
    return ('F1',score,False)


#重新把train-df和test_df分开
train_df = df_all.loc[train_df.index]
test_df.loc[test_df.index]



'''
试试神经网络
拓扑结构：
第一层：162个输入
第二层：16个神经元，使用sigmoid激活函数
第三层：8个神经元，使用sigmoid激活函数
第四层：1个神经元，使用sigmoid激活函数
优化函数：使用Adam优化函数，根据情况使用正则化
'''
import tensorflow as tf
from tensorflow.python.framework import ops

def create_placeholders(n_x, n_y):
    
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])
    
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)
    ### END CODE HERE ###
    
    return X, Y, keep_prob_1, keep_prob_2

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [16, 162]
                        b1 : [16, 1]
                        W2 : [8, 16]
                        b2 : [8, 1]
                        W3 : [1, 8]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [16, 162], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [16,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [8, 16], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [8,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [2, 8], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [2,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters, keep_prob_1, keep_prob_2):
    """
    Implements the forward propagation for the model: LINEAR -> sigmoid -> LINEAR -> sigmoid -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    
    Z1 = tf.add(tf.matmul(W1, X), b1)         # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.sigmoid(Z1)                    # A1 = relu(Z1)
    A1_drop = tf.nn.dropout(A1, keep_prob_1)
    Z2 = tf.add(tf.matmul(W2, A1_drop), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.sigmoid(Z2)                     # A2 = relu(Z2)
    A2_drop = tf.nn.dropout(A2, keep_prob_2)
    Z3 = tf.add(tf.matmul(W3, A2_drop), b3)    # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
#    cost = -tf.reduce_sum(labels*tf.log(logits) + (1 - labels) * tf.log(1 - logits))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y, keep_prob_1, keep_prob_2 = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters, keep_prob_1, keep_prob_2)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob_1:0.75, keep_prob_2:1})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3, 0),tf.argmax(Y, 0))
        y_ = tf.argmax(Z3, 0)
        y = tf.argmax(Y, 0)
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob_1:0.75, keep_prob_2:1}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob_1:1, keep_prob_2:1}))
        
        
        return parameters



X_train, X_test, Y_train, Y_test = train_test_split(train_df.values, train_Y.values ,test_size=0.2, random_state=0)

X_train = X_train.T
X_test = X_test.T

Y_train = convert_to_one_hot(Y_train, 2)
Y_test = convert_to_one_hot(Y_test, 2)

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


parameters = model(X_train, Y_train, X_test, Y_test)


