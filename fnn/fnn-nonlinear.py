# -*- coding: utf-8 -*-

# @Author: zhoujiamu
# @Time:  2019/3/22 上午7:45


import tensorflow as tf
from numpy.random.mtrand import RandomState

batch_size = 128

# 定义权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
biases1 = tf.Variable(tf.constant(0.01, shape=(1, 3)))
# biases1 = tf.Variable([[0.7, 0.8, 0.9], [0.1, 0.1, 0.1]])

w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))
biases2 = tf.Variable(tf.constant(0.01, shape=(1, 1)))

# 定义输入
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

a = tf.sigmoid(tf.matmul(x, w1) + biases1)

y = tf.tanh(tf.matmul(a, w2) + biases2)

# 定义损失函数

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 定义优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)

dataset_size = 1000000

X = rdm.rand(dataset_size, 2)

Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1 + biases1))

    print("init value: ")
    print("layer1: ")
    print(sess.run(w1))
    print(sess.run(biases1))
    print("layer2: ")
    print(sess.run(w2))
    print(sess.run(biases2))

    # 设定训练轮数
    STEPS = 10000

    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross_entropy on all data is %g" % (i, total_cross_entropy))

    print("finished train weight: ")
    print("layer1: ")
    print(sess.run(w1))
    print(sess.run(biases1))
    print("layer2: ")
    print(sess.run(w2))
    print(sess.run(biases2))




