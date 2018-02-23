import tensorflow as tf

# implement XOR NN machine using multi layer perceptron
def relu_tf(x):
    return tf.maximum(x, 0.0)

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [0, 1, 1, 1]

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# define NN function model
## define hidden layer
W11 = tf.Variable(tf.zeros([2, ]))
b11 = tf.Variable(tf.zeros([1, ]))
L11 = relu_tf(tf.add(tf.reduce_sum(tf.multiply(X, W11), 1), b11))

W12 = tf.Variable(tf.zeros([2, ]))
b12 = tf.Variable(tf.zeros([1, ]))
L12 = relu_tf(tf.add(tf.reduce_sum(tf.multiply(X, W12), 1), b12))

## define out layer
W2 = tf.Variable(tf.zeros([2, ]))
b2 = tf.Variable(tf.zeros([1, ]))
L2 = tf.concat([tf.expand_dims(L11, -1), tf.expand_dims(L12, -1)], 1)
hypothesis = tf.add(tf.reduce_sum(tf.multiply(L2, W2), 1), b2)

# define optimization model
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# run NN
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # set learning iteration parameter
    #   - iteration count
    #   - error value
    #   - based on normal distribution?
    #   - error change rate (no more changed rapiddly stop) found at tensor0004.py
    for step in range(10000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if (step + 1) % 10 == 0:
            print(step + 1, cost_val, sess.run(hypothesis, feed_dict={X: x_data}))

    # testing
    print("\n=== Test ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))
