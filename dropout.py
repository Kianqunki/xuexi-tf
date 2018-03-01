# -*- coding:utf-8 -*-
#
#tensorflow中的dropout就是:
# 使输入tensor中某些元素变为0，
# 其它没变0的元素变为原来的1/keep_prob大小！
# 目标是强迫神经网络学习更多知识

import tensorflow as tf
keep_prob = tf.placeholder(tf.float32)
x = tf.Variable(tf.ones([10,10]))
y = tf.nn.dropout(x, keep_prob)
s = tf.reduce_sum(y)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(s, feed_dict = { keep_prob: 0.4})
