# -*- coding:utf-8 -*-
#实验数据的构造
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, batch_size, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

#-------------------------------------------------------------------------------
# 定义模型
def rnn_network(num_steps, state_size, batch_size=200, learning_rate=0.1):
    num_classes = 2
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
    #-------------------------------------------------------------------------------------------------------

    x_one_hot = tf.one_hot(x, num_classes)
    rnn_input_list = tf.unstack(x_one_hot, axis=1)

    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

    def rnn_cell(rnn_input, state):
        with tf.variable_scope('rnn_cell', reuse=True):
            W = tf.get_variable('W', [num_classes + state_size, state_size])
            b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
        return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

    init_state = tf.zeros([batch_size, state_size])
    state = init_state
    rnn_output_list = []
    for rnn_input in rnn_input_list:
        state = rnn_cell(rnn_input, state)
        rnn_output_list.append(state)
    final_state = rnn_output_list[-1]

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_output_list]
    predictions_list = [tf.nn.softmax(logit) for logit in logits]

    y_as_list = tf.unstack(y, num=num_steps, axis=1)

    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for
            logit, label in zip(logits, y_as_list)]
    mean_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(mean_loss)

    def train_network(num_epochs, verbose=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            for idx, epoch in enumerate(gen_epochs(num_epochs, batch_size, num_steps)):
                training_loss = 0
                training_state = np.zeros((batch_size, state_size))
                if verbose:
                    print "\nEPOCH", idx
                for step, (X, Y) in enumerate(epoch):
                    tr_losses, training_loss_, training_state, _ = sess.run([losses,
                        mean_loss, final_state, train_step],
                        feed_dict={x:X, y:Y, init_state:training_state})
                    training_loss += training_loss_
                    if step % 100 == 0 and step > 0:
                        if verbose:
                            print "Average loss at step", step, "for last 250 steps:", training_loss/100
                        training_losses.append(training_loss/100)
                        training_loss = 0
        return training_losses
    return train_network
#-------------------------------------------------------------------------------------------------------
train = rnn_network(10, 16)
plt.plot(train(1))
#-------------------------------------------------------------------------------------------------------
print "理想的交叉熵:"
print "- 没有学习到任何规律: ",\
        -(0.625 * np.log(0.625) + 0.375 * np.log(0.375)),\
        0.625
print "- 只学习到第一条规律: ",\
        - 0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))\
        - 0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)),\
        0.875*0.5+0.625*0.5
print "- 两条规律都学习成功: ",\
        - 0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))\
        - 0.25 * (2 * 0.50 * np.log (0.50))\
        - 0.25 * (0),\
        0.75*0.5+0.5*0.25+1*0.25
#-------------------------------------------------------------------------------------------------------
plt.show()
