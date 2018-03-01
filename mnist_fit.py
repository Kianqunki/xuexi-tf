# -*- coding:utf-8 -*-
# 采用拟合方式,对MNIST数据进行数字图像识别
#
# mnist 包含三个数据集
# .train 55000 个训练用点数据集
# .test  10000 个测试数据集 
# .validation 5000 个验证数据集
# 每个数据集分为
#       .images  象素28×28=784
#       .lables  对应数字[0,9]10种分类
#
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf
# 每行代表一幅图
x = tf.placeholder(tf.float32, [None, 784]) # None表示可取任意值

# 每个象素相对[0,9]的权重系数
W = tf.Variable(tf.zeros([784, 10]))
# [0,9]的权值偏置量,用于拟合
b = tf.Variable(tf.zeros([10]))

# 每行对应[0,9]的相应权值(越大越可能),
# 但每行和不为1, 即sum(p[i]) != 1 因此权值并不是概率
p = tf.matmul(x,W) + b #加法非严格,会进行自然的转换,结果形状是[None,10]

# softmax 模型可以用来给不同的对象分配概率
#   y[i, j] = exp(p[i, j]) / sum(exp(p[i]))
# 每行图中对应[0,9]的概率
y = tf.nn.softmax(p)#结果形状是[None,10]

# y 是所预测的概率分布,y_是真实的分布(即标签, 单个1的向量)
y_ = tf.placeholder(tf.float32, [None, 10])

# 成本函数"交叉熵", 尽量最小化这个指标
# 当y, y_相等时取到极大值,故取反得极小值
cross_entropy = - tf.reduce_sum(y_ * tf.log(y)) # reduce_sum求张量中所有元素的和

# 梯度下降算法,以 0.01 的学习速率最小化 cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 进行训练
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 每次取100行图
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

#模型的评价
# tf.arg_max(y, 1) 返回第1维度的最大值索引,即最大概率的数值[0,9]
# 返回bool型强转为数值型,结果形状是[None, 1]. 每行, 1表示正确,0表示错误
correct_prediction = tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)), tf.float32) 
accuracy = tf.reduce_mean(correct_prediction) # reduce_mean 求平均值

print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
