import tensorflow as tf

def step_fn(x):
    return tf.cast(x > 0, tf.float32)

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

X = tf.placeholder(tf.float32, name="X")

# define NN function model
W = tf.Variable(tf.fill([2, ], 0.5))
b = tf.constant(-0.7)
#hypothesis = tf.reduce_sum(tf.multiply(X, W), 1) + b
hypothesis = step_fn(tf.reduce_sum(tf.multiply(X, W), 1) + b)

# using step function, calculate AND gate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(hypothesis, feed_dict={X: x_data, }))
