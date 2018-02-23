import tensorflow as tf
import numpy as np
import tensormodel as nn

# implement OR NN machine using tensorflow

# define input pattern
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# define output pattern, rank of output pattern should be equal to input, or tensorflow broadcast output tensor and result is unpredictable
y_data = np.array([[0], [1], [1], [1]])

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

with tf.name_scope('output'):
    # define NN function model
    ## W <- weight of connection
    ## b <- bias of neuron
    ## It is important to define initial value for Neuron network to find solution
    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='Weight')
    nn.variable_summaries(W, 'Weight')

    ## bias is disabled until general delta rule is used
    # b = tf.Variable(tf.zeros([1, ]))
    b = tf.constant(0, dtype=tf.float32, shape=[1, ], name='Bias')

    #hypothesis = tf.reduce_sum(tf.multiply(X, W), 1) + b
    ## hypothesis is active value from neuron
    ## X is active value of input neuron
    ## W is weight of connection from that neuron
    #hypothesis = nn.sigmoid_tf(tf.reduce_sum(tf.multiply(X, W), 1) + b)
    hypothesis = nn.sigmoid_tf(tf.add(tf.matmul(X, W), b))
    nn.variable_summaries(hypothesis, 'Active')


with tf.name_scope('optimizer'):
    # define optimization model
    ## at the first version, I consider only recall process and missed learning process, chaning weight of NN
    ## learning_rate <- Hebbian rule
    # cost = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y))))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(cost)
    nn.variable_summaries(cost, 'Cost')


# run NN
with tf.Session() as sess:
    ## initialize neuron network
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    print(sess.run(W), sess.run(b))
    #print(sess.run(tf.matmul(X, W), feed_dict={X: x_data}))
    #print(sess.run(hypothesis, feed_dict={X: x_data}))
    print(sess.run(tf.subtract(hypothesis, Y), feed_dict={X: x_data, Y: y_data}))
    #print(sess.run(tf.square(tf.subtract(hypothesis, Y)), feed_dict={X: x_data, Y: y_data}))
    #print(sess.run(tf.reduce_sum(tf.square(tf.subtract(hypothesis, Y))), feed_dict={X: x_data, Y: y_data}))
    #print(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    grad = optimizer.compute_gradients(cost)
    print(sess.run(grad, feed_dict={X: x_data, Y: y_data}))
    sess.run(optimizer.apply_gradients(grad), feed_dict={X: x_data, Y: y_data})
    print(sess.run(W))

    # set learning iteration parameter
    #   - iteration count
    #   - error rate
    #   - based on normal distribution?
    for step in range(500):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, step)

        if (step + 1) % 100 == 0:
            print(step + 1, cost_val, sess.run(W), sess.run(b))

    # testing
    print("\n=== Test OR Implement ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))
