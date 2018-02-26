import tensorflow as tf
import numpy as np
import tensormodel as nn

# implement AND NN machine using tensorflow

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [[0], [0], [0], [1]]

X = tf.placeholder(tf.float32, [None, 2], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

layer = nn.NnFullConnectedLayer(X, 2, 1, tf.sigmoid_tf, bias='Variable')
learn = nn.GradientDescentOptimizer(Y, layer)

run = nn.RunLearnModel("AND")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})

print("\n=== Test AND Implement ===")
num_data = len(x_data)
for i in range(num_data):
    x_i = x_data[i:i +1] # shape of x_data[0] and x_data[:] (slice) is different
    y_i = y_data[i:i +1]
    print("I:{}, O:{}/R:{}".format(x_i, y_i, run.Recall(layer, feed_dict={X:x_i, Y:y_i})))
