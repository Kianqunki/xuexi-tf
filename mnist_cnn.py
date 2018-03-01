# -*- coding:utf-8 -*-
# 采用卷积神经网络方式,对MNIST数据进行数字图像识别
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
#每行代表一幅图
x = tf.placeholder(tf.float32, [None, 784]) # None表示可取任意值

#实用函数
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
 
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    '''
二维卷积

x：输入多张图片,形状为[图厚,图高,图宽,输入通道数]
   如果是普通灰度照片,输入通道数=1,如果是RGB彩色照片, 输入通道数=3

W: 多层卷积核,形状为[核高,核宽,输入通道数,核层数]
   核高和核宽指的是本次卷积计算的“抹布”的规格,输入通道数要一致

return: 形状为[图厚,图高,图宽,核层数]
'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
最大值池化
x：输入多张图片,形状为[图厚,图高,图宽,输入通道数]
   如果是普通灰度照片,输入通道数=1,如果是RGB彩色照片, 输入通道数=3
return: 形状为[图厚,图高/2,图宽/2,输入通道数]
'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First Convolutional Layer | 第一层卷积
# 卷积+池化

# 卷积
#   在每个5×5 的 patch 中算出 32 个特征(32层卷积核).
#   卷积核是一个[5,5,1,32]的张量,前两个维度是 patch 的大小,
#   接着是输入的通道数目,最后是输出的通道数目.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 为了用这一层,我们把x变成一个4维张量,第2,3 维对应图片的宽高,最后一维代表颜色通道。
x_image = tf.reshape(x, [-1,28,28,1])#至多一个-1,表示自动计算此维度,结果形状是[None,28,28,1]

# 把x_image和权值向量进行卷积相乘,加上偏置,使用 ReLU 激活函数,最后池化
h_conv1 = conv2d(x_image, W_conv1) #结果形状是[None,28,28,32], 相对原始图增大32倍

# relu()将所有负元素置0
h_relu1 = tf.nn.relu(h_conv1 + b_conv1) 
h_pool1 = max_pool_2x2(h_relu1)#结果形状是[None,14,14,32],相对原始图增大8倍

# Second Convolutional Layer | 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(h_pool1, W_conv2) #结果形状是[None,14,14,64],相对原始图增大16倍
h_relu2 = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = max_pool_2x2(h_relu2)#结果形状是[None,7,7,64],相对原始图增大4倍

# Densely Connected Layer | 密集连接层
# 图片降维到 7 x 7,我们加入一个有 1024 个神经元的全连接层,用于处理整个图片
# 我们把池化层输出的张量 reshape 成一些向量,乘上权重矩阵,加上偏置,使用 ReLU 激活
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])#每行为一幅图
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#结果形状是[None,1024]
# 将784的原图通过卷积+池化,提取1024个特征值

# keep_prob 是元素保持概率(即乘1/keep_prob),反之变0
# 但保持reduce_sum()不变,即 reduce_sum(h_fc1_drop) == reduce_sum(h_fc1)
# 目的是丢掉一些随机特征值,强迫神经网络学习更多知识
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer | 输出层
# h_fc1_drop 形状为[None,1024]
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 每行对应[0,9]的相应权值(越大越可能),
# 但每行和不为1, 即sum(p[i]) != 1 因此权值并不是概率
p = tf.matmul(h_fc1_drop,W_fc2) + b_fc2 #加法非严格,会进行自然的转换,结果形状是[None,10]


# softmax 模型可以用来给不同的对象分配概率,每列是某类的权值
#   y_conv[i, j] = exp(p[i, j]) / sum(exp(p[i]))
# 每行图中对应[0,9]的概率
y_conv=tf.nn.softmax(p)

#Train and Evaluate the Model | 训练和评估模型
# y_conv 是所预测的概率分布,y_是真实的分布(即标签, 单个1的向量)
y_ = tf.placeholder(tf.float32, [None, 10])

# 成本函数"交叉熵", 尽量最小化这个指标
# 当y_conv, y_相等时取到极大值,故取反得极小值
cross_entropy = tf.reduce_sum(-y_ * tf.log(y_conv)) # reduce_sum求张量中所有元素的和
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#模型的评价
# tf.arg_max(y_conv, 1) 返回第1维度的最大值索引,即最大概率的数值[0,9]
# 返回bool型强转为数值型,结果形状是[None, 1]. 每行, 1表示正确,0表示错误
correct_prediction = tf.cast(tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1)), tf.float32) 
accuracy = tf.reduce_mean(correct_prediction) # reduce_mean 求平均值

# 进行训练
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 每次取100行图
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={ x:batch_xs, y_:batch_ys, keep_prob:1.0 })
        print "step %d, training accuracy %g"%(i, train_accuracy)
    sess.run(train_step, feed_dict={ x:batch_xs, y_:batch_ys, keep_prob:0.5 })

saver.save(sess, 'mnist_cnn')
# 测试评估
print "Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
