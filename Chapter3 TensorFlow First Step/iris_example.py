# encoding: utf-8
"""
@author: nnn11556
@software: PyCharm
@file: iris_example.py
@time: 2018/11/5 16:18
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

#read data and preprocessing
IRIS_PATH = r"IRIS_data\iris.csv"
iris = pd.read_csv(os.path.join(os.getcwd(),IRIS_PATH),header=None)
#code species to number 1-3
iris[5] = pd.Categorical(iris[5]).codes
#read x data
iris_x = iris.iloc[:,1:5].as_matrix()[1:]
iris_y = iris[5].as_matrix()[1:]
#read y data and expand to 2D array
iris_y = np.atleast_2d(iris_y).T
#one hot code
ohe = OneHotEncoder()
ohe.fit(iris_y)
iris_y = ohe.transform(iris_y).toarray()
#split to test set and train set
x_train,x_test,y_train,y_test=train_test_split(iris_x,iris_y,test_size=0.25)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None,4])
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))
#softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 3])
#cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#GDoptimizer learning rate = 0.1 loss = cross entropy
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3000):
        # randomly select 5 sample from train set to train the model(SGD)
        k = np.random.randint(0,int(150*0.75)-5)
        batch_xs, batch_ys = np.atleast_2d(x_train[k:k+5]),np.atleast_2d(y_train[k:k+5])
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
    #cheak model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test score:',sess.run(accuracy,feed_dict={x:x_test, y_:y_test}))




