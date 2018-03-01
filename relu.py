import tensorflow as tf
x = tf.constant([1,-1,100,-10, -0.1, 0.1]);
y = tf.nn.relu(x)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(y)
