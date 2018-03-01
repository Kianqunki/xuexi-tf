# -*- coding:utf-8 -*-

import tensorflow as tf 

#####################################################################################
#从这行开始，TensorFlow已经开始管理许多状态，它有一个隐式默认的图，
#_存在目录default_graph_stack下，但是我们不能直接访问，而是使用tf.get_default_graph()
graph = tf.get_default_graph()

#TensorFlow图的节点称为"operation操作"或“ops”
print graph.get_operations()
#目前在图中什么也没有，我们需要放入需要让TensorFlow计算的数据，开始放入一个简单常量值

input_value = tf.constant(1.0, name='input')
#放入一个简单常量值

operations = graph.get_operations()
print operations
#TensorFlow内部使用protocol buffer，它是一种类似JSON的格式，
#打印出常量操作的node_def可以显示出TensorFlow第一个的protocol buffer视图

#TensorFlow分离了计算的定义和计算的执行，它们可以分别在不同地方进行，
#一个图定义了操作，而操作只会在一个会话session中发生，图和会话是独立创建的，
#图类似计划，而会话类似计划的实施。这种方式被称为懒赋值或懒计算

#为了确实获得input_value的数值，我们需要创建一个会话session，
#在这个会话中，图操作才能真正被计算，这需要我们显式明确命令其运行一次
sess = tf.Session() 
print sess.run(input_value)
print '#############################################################################'

# 神经元权重不是一个常量，因为在以后训练过程中会依据输出情况不断调整，
# 权重是一个TensorFlow的变量，我们首先从值0.8开始
weight = tf.Variable(0.8, name='weight')
#你也许以为增加一个变量会增加一个操作到图中，但是实际上一行代码会增加四个操作，
#我们看看所有操作名称
print "ops:"
for op in graph.get_operations(): print "\t", op.name

output_value = weight * input_value
#现在图中有六个操作了，最后一个是乘法
op = graph.get_operations()[-1]
print op.name, "inputs:"
for op_input in op.inputs: print "\t", op_input

#TensorBoard是通过查看一个TensorFlow会话创建的输出的目录来工作的。
#我们可以先用一个SummaryWriter来写这个输出。如果我们只是创建一个图的输出，它就将图写出来。
#构建SummaryWriter的第一个参数是一个输出目录的名字。如果此目录不存在，则在构建SummaryWriter时会被建出来。
summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
#现在我们可以通过命令行来启动TensorBoard了。
#$ tensorboard –logdir=log_simple_graph
#TensorBoard会运行一个本地的Web应用，端口6006(6006是goog这个次倒过的对应).
#在你本机的浏览器里登陆localhost:6006/#graphs，你就可以看到在TensorFlow里面创建的图
print '#############################################################################'

#需要运行run获得output_value 结果值，但是这个操作会依赖weight权重，而权重初始值是0.8，
#但是我们没有在当前会话设置这个初始值，
#tf.initialize_all_variables()会帮助我们初始所有的变量初始值
init = tf.initialize_all_variables() 
sess.run(init)

#现在我们可以运行output_value 操作
print(sess.run(output_value))
print '#############################################################################'

#让我们假设设置输入值是1.0，而正确输出值是0，我们也有一个带有一个特征的简单训练数据集，
#只有一个值和一个标签label，它是零，我们让神经元学习从1到0的函数

#目前的系统是获得输入1，返回的是0.8，这是不正确的，
#我们需要一个办法衡量系统是怎么错的
#我们称为这种错误衡量是"loss"，

#设定我们系统一个最小的loss，如果loss是负数，再减小就愚蠢了

#让我们在当前输出和期望输出之间差值做个平方
y_ = tf.constant(0.0)
loss = (output_value - y_)**2

#现在我们需要一个优化器，我们使用一个梯度递减的优化器，
#这样能够基于loss不断修改权重，

#这个优化器会采用一个学习率来调整更新的大小幅度，这个学习率设置为0.025

optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)

#优化器是非常聪明，它能自动工作，在整个网络中使用合适的梯度，实施后续的学习步骤。
#对于我们这个简单案例我们看看梯度是什么样

grads_and_vars = optim.compute_gradients(loss)
sess.run(tf.initialize_all_variables()) #加入了新变量, 因此还要运行一次
print(sess.run(grads_and_vars))
#为什么梯度的值是1.6, 我们的损失函数是错误的平方，因此它的导数就是这个错误乘2.
#现在系统的输出是0.8而不是0，所以这个错误就是0.8，乘2就是1.6。优化器是对的!
print '#############################################################################'

#对于更复杂的系统，TensorFlow可以自动地计算并应用这些梯度值。
#让我们运用这个梯度来完成反向传播
sess.run(optim.apply_gradients(grads_and_vars))
print(sess.run(weight))
#现在权重减少了0.04，这是因为优化器减去了梯度乘以学习比例（1.6*0.025
#权重向着正确的方向在变化
print '#############################################################################'

#其实我们不必像这样调用优化器.我们可以形成一个运算，自动地计算和使用梯度：train_step
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(100): sess.run(train_step)
print(sess.run(weight))
#多次运行训练步骤后，权重和输出值已经非常接近0了。这个神经元已经学会了！
