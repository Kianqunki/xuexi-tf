# -*- coding:utf-8 -*-
# tf.unstack(value,num=None,axis=0,name='unstack')　　
# 将给定的R维张量拆分成R-1维张量
# 将value根据axis分解成num个张量，返回的值是list类型，如果没有指定num则根据axis推断出
import tensorflow as tf
a = tf.constant([[3,2],[4,5],[6,7]])
with tf.Session() as sess:
    print '原始数据',a.get_shape(),':'
    print sess.run(a)
    print '--------'
    print 'unstack axis=0'
    print sess.run(tf.unstack(a,axis=0))
    print '--------'
    print 'unstack axis=1'
    x = tf.unstack(a,axis=1)
    print sess.run(x)
    print sess.run(tf.stack(x, axis=1))
