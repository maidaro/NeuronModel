import tensorflow as tf

def relu_tf(x):
    return tf.maximum(x, 0.0)

# implement OR NN machine relu function

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [0, 1, 1, 1]

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# define NN function model
W = tf.Variable(tf.fill([2], 0.0))
b = tf.Variable(tf.zeros([1]))
#hypothesis = tf.reduce_sum(tf.multiply(X, W), 1) + b
hypothesis = relu_tf(tf.reduce_sum(tf.multiply(X, W), 1) + b)
#hypothesis = tf.nn.relu(tf.reduce_sum(tf.multiply(X, W), 1) + b)

# define optimization model
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# run NN
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # set learning iteration parameter
    #   - iteration count
    #   - error value
    #   - based on normal distribution?
    #   - error change rate (no more changed rapiddly stop) found at tensor0004.py
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if (step + 1) % 10 == 0:
            print(step + 1, cost_val, sess.run(W), sess.run(b))

    # testing
    print("\n=== Test ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))

# 1019 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1020 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1021 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1022 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1023 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1024 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1025 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1026 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1027 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1028 0.003906251 [0.4999988 0.4999988] [0.25000137]
# 1029 0.003906251 [0.4999988 0.4999988] [0.25000137]

# === Test ===
# X: [0, 0], Y: [0.25000137]
# X: [0, 1], Y: [0.7500002]
# X: [1, 0], Y: [0.7500002]
# X: [1, 1], Y: [1.249999]
# [Finished in 8.9s]
