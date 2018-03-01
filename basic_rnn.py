# -*- coding:utf-8 -*-
#实验数据的构造
#   输入数据X：在时间t，X[t]的值有50%的概率为1，50%的概率为0；
#   输出数据Y：在时间t，Y[t]的值有50%的概率为1，50%的概率为0;
#           除此之外,有两条规律
#           1.如果`X[t-3] == 1`，Y[t]为1的概率增加50%,即100%为1
#           2.如果`X[t-8] == 1`，Y[t]为1的概率减少25%,即 25%为1
#
#   如果上述两个条件同时满足，则Y[t]为1的概率为75%
#   可知，Y与X有两个依赖关系，一个是t-3，一个是t-8
#   我们实验的目的就是检验RNN能否捕捉到Y与X之间的这两个依赖关系。实验使用交叉熵作为评价标准
#
#交叉熵定义
#   一个样本的所需要的编码长度的期望(即平均编码长度)为：H(p)=sum(-p[i]*log(p[i])。
#   如果使用观测分布q来表示来自真实分布p的平均编码长度，则应该是：H(p,q)=sum(-p[i]*log(q[i]))
#   因为用q来编码的样本来自分布p，所以期望H(p,q)中概率是p(i)。H(p,q)我们称之为“交叉熵”。
#
#   比如含有4个字母(A,B,C,D)的数据集中，真实分布p=(1/2, 1/2, 0, 0),即A和B出现的概率均为1/2，C和D出现的概率都为0。
#   计算H(p)=1，即只需要1位编码即可识别A和B。
#   如果使用分布q=(1/4, 1/4, 1/4, 1/4)来编码则得到H(p,q)=2，即需要2位编码来识别A和B(当然还有C和D，尽管C和D并不会出现)
#
#三条理想的实验结果
#   Y值采样(1,0),对应的真实分布p=(0.625,0.375), 其中0.625=0.5+0.5*0.5-0.5*0.25
#   则判断Y[t]为1的正确率采样(对,错),与Y[t]值采样(1,0),概率是等价的.
#   1. 若没有学习到任何一条规律,正确率分布q=(0.625,0.375)
#      交叉熵0.66=-(0.625 * np.log(0.625) + 0.375 * np.log(0.375))
#   2. 若学习到条件1，即X[t-3]为1时Y[t]一定为1,由于潜在条件2, 0.5*(1-75%)的错误率
#      交叉熵0.52=(-0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125)) -0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
#   3. 若学会两条依赖,通过判断{X[t-3],X[t-8]}的4种组合
#      - 当{1,0}时能确定正确,  因此0.25概率正确率是100%
#      - {1,1}{0,1}两种组合时Y[t]为1的概率为75%, 因此0.5 概率正确率是75%
#      - {0,0}时,还有0.25的概率正确率是50%。所以其交叉熵为
#      0.45 (-0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25)) - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#实用函数
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
#-------------------------------------------------------------------------------------------------------
# 生成实验数据
def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        #判断X[i-3]和X[i-8]是否为1，修改阈值
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        #生成随机数，以threshold为阈值给Yi赋值
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

#遍历单次实验数据raw_data
#遍历方式:
#   每行batch_size
#   每列num_steps
# 因此返回的元素形状是[batch_size,num_steps]
def gen_batch(raw_data, batch_size, num_steps):
    #raw_data是使用gen_data()函数生成的数据，分别是X和Y
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 首先将数据平均分成batch_size份，data_x[i],data_y[i] 代表第i份
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

    #因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。 
    epoch_size = batch_partition_length // num_steps

    #x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, batch_size, num_steps):
    """ 构造训练数据并遍历
    
    共两重遍历: 先按训练次数,每次返回数据生成器.
    数据生成器再按形状 [batch_siza,num_steps] 遍历数据

    Args:
        n: 训练次数,每次返回数据生成器
        batch_size: 数据生成器中每次生成的行数
        num_steps: 数据生成器中每次生成的列数

    Returns:
        返回数据生成器
    """
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

def static_rnn(rnn_input_list, init_state):
    """ 创建单层静态RNN图

    单层的RNN公式:

        S[t]=tanh((X[t], S[t−1])*W+b) 
        L[t]=S[t]*U+d

    Args:
        rnn_input_list: 输入数据list
        init_state: 初始状态

    Returns:
        logit_list:  对应的预测概率list
        final_state: 最终输出状态

    """
    num_classes = int(rnn_input_list[0].get_shape()[-1])
    state_size = int(init_state.get_shape()[-1])

    # 方案2: 直接使用库函数
    # cell = tf.contrib.rnn.BasicRNNCell(state_size)
    # rnn_output_list, final_state = tf.contrib.rnn.static_rnn(cell, rnn_input_list, initial_state=init_state)

    # 方案3: TODO
    # 使用dynamic_rnn函数，动态构建RNN模型
    # rnn_output_list, final_state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=init_state)

    W = weight_variable([num_classes+state_size,state_size])
    b = bias_variable([state_size])

    state = init_state
    rnn_output_list = []
    for rnn_input in rnn_input_list:
        state = tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
        rnn_output_list.append(state)

    final_state = rnn_output_list[-1]

    # 将rnn_output_list转为logit_list (抽象的权值)
    # 注意，这里要将num_steps个输出全部分别进行计算其输出
    # logit_list 为list,每个元素形状为[batch_size, num_classes]
    U = weight_variable([state_size, num_classes])
    d = bias_variable([num_classes])
    logit_list = [tf.matmul(rnn_output, U) + d for rnn_output in rnn_output_list]

    return logit_list, final_state
#-------------------------------------------------------------------------------------------------------
# 定义模型
def rnn_network(num_steps, state_size, batch_size=200, learning_rate=0.1):
    #---------------------------------------------------------------------------
    # 输入数据
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = tf.placeholder(tf.int32, [batch_size, num_steps])
    y = tf.placeholder(tf.int32, [batch_size, num_steps])
    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    #---------------------------------------------------------------------------
    # 整理数据
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # x,y的元素为输入值(0或1),转化为one-hot编码的张量,只需两个类别
    num_classes = 2

    # x_one_hot 形状为 [batch_size, num_steps, num_classes]
    x_one_hot = tf.one_hot(x, num_classes)

    # 将输入unstack，即在num_steps上解绑，方便给每个循环单元输入。
    # 这里可以看出RNN每个cell都处理一个batch的输入（即batch个样本输入）
    # rnn_input_list 是list, 长度为num_steps, 每个元素形状为 [batch_size, num_classes]
    rnn_input_list = tf.unstack(x_one_hot, axis=1)

    #---------------------------------------------------------------------------
    # 静态RNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # logit_list为list,长度为num_steps, 每个元素形状为[batch_size, num_classes]
    # 对应rnn_input_list元素的预测权值
    # final_state为末尾元素
    logit_list, final_state = static_rnn(rnn_input_list, init_state)

    #---------------------------------------------------------------------------
    # 交叉熵评估
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # y是真实的分布,在num_steps上解绑,y_as_list是list,每个元素形状为[batch_size]
    y_as_list = tf.unstack(y, num=num_steps, axis=1)

    # 所有num_steps个交叉熵,cross_entropy是其平均值
    cross_entropy_list =[
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
            for logit, label in zip(logit_list, y_as_list)]
    cross_entropy = tf.reduce_mean(cross_entropy_list)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

    #---------------------------------------------------------------------------
    # 模型评价
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 使用softmax 给不同的对象分配概率(归一化),每列是某类的权值
    # predictions_list是list, 长度num_steps, 每个元素形状为[batch_size, num_classes]
    #predictions_list = [tf.nn.softmax(logit) for logit in logits]

    # predictions形状为[batch_size, num_steps, num_classes]
    #predictions = tf.stack(predictions_list, axis=1)

    #correct_prediction = tf.cast(tf.equal(tf.argmax(predictions, 2), tf.cast(y,tf.int64)), tf.float32) 
    #accuracy = tf.reduce_mean(correct_prediction)

    #---------------------------------------------------------------------------
    # 训练函数
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train(num_epochs, verbose=True):
        training_cross_entropy = []#训练过程中的交叉熵
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for idx, epoch in enumerate(gen_epochs(num_epochs, batch_size, num_steps)):
                training_state = np.zeros((batch_size, state_size))
                cross_entropy_sum = 0
                if verbose:
                    print "\n训练次数", idx
                for step, (X, Y) in enumerate(epoch):
                    training_loss_, training_state, _ = sess.run(
                            [cross_entropy, final_state, train_step],
                            feed_dict={x:X, y:Y, init_state:training_state})
                    cross_entropy_sum += training_loss_
                    if step % 100 == 0 and step > 0:
                        training_cross_entropy.append(cross_entropy_sum/100)
                        cross_entropy_sum = 0
                        if verbose:
                            print "Average loss at step", step, "for last 250 steps:", training_cross_entropy[-1]
        
        return training_cross_entropy

    return train
#-------------------------------------------------------------------------------------------------------
train = rnn_network(10, 16)
plt.plot(train(10))
#-------------------------------------------------------------------------------------------------------
print "理想的交叉熵:"
print "- 没有学习到任何规律: ",\
        -(0.625 * np.log(0.625) + 0.375 * np.log(0.375))
print "- 只学习到第一条规律: ",\
        - 0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))\
        - 0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375))
print "- 两条规律都学习成功: ",\
        - 0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))\
        - 0.25 * (2 * 0.50 * np.log (0.50))\
        - 0.25 * (0)
#-------------------------------------------------------------------------------------------------------
#plt.show()
