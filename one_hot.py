# -*- coding:utf-8 -*-
# tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
import tensorflow as tf  
indices = [0, 2, -1, 1]
depth = 3
with tf.Session() as sess:
    print sess.run(tf.one_hot(indices, depth))

on_value = 5.0
off_value = 0.0
axis = -1
