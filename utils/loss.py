# -*- coding: utf-8 -*-

# @Author: zhoujiamu
# @Time:  2019/3/23 下午10:40

# 自定义损失函数

import tensorflow as tf


def cross_entropy(p, q):
    return -tf.reduce_sum(p * tf.log(q), axis=1)


if __name__ == '__main__':

    with tf.Session() as sess:
        x0 = tf.constant([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        x1 = tf.constant([[0.1, 0.7, 0.3, 0.2], [0.3, 0.6, 0.5, 0.9]])

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=x0, logits=x1)

        loss1 = cross_entropy(x0, tf.nn.softmax(x1))

        sess.run(tf.global_variables_initializer())

        print(sess.run(loss))
        print(sess.run(loss1))

        print(sess.run(tf.nn.softmax(x1)))

