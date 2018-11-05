# encoding: utf-8
"""
@author: nnn11556
@software: PyCharm
@file: Softmax_Regression.py
@time: 2018/10/29 15:40
"""
import os,sys
#载入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'MNIST_data/', one_hot=True)
#MNIST数据集相关参数
print('train data set :image {0},label {1}'.format(mnist.train.images.shape, mnist.train.labels.shape))
print('test data set :image {0},label {1}'.format(mnist.test.images.shape, mnist.test.labels.shape))
print('validation data set:image {0},label {1}'.format(mnist.validation.images.shape, mnist.validation.labels.shape))

import tensorflow as tf
x = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#softmax regression定义
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
    #在测试集上验证
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
