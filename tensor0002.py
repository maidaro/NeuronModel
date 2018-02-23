import tensorflow as tf
import numpy as np
import tensormodel as nn

# implement AND NN machine using tensorflow

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [[0], [0], [0], [1]]

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# define NN function model
W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='Weight')
b = tf.constant(0, dtype=tf.float32, shape=[1, ], name='Bias')
#hypothesis = tf.reduce_sum(tf.multiply(X, W), 1) + b
hypothesis = nn.sigmoid_tf(tf.add(tf.matmul(X, W), b))

# define optimization model
cost = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# run NN
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # set learning iteration parameter
    #   - iteration count
    #   - error rate
    #   - based on normal distribution?
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if (step + 1) % 100 == 0:
            print(step + 1, cost_val, sess.run(W), sess.run(b))

    # testing
    print("\n=== Test ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))
