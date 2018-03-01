# -*- coding:utf-8 -*-

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

import cv2
net = cv2.dnn.readNetFromTensorflow("mnist_cnn")
print type(mnist.test.images)
net.setInput("x", mnist.test.images);
net.setInput("keep_prob", 1.0);
net.setInput("y_", mnist.test.labels);
net.forward("accuracy")
