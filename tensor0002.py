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

output = nn.NnFullConnectedLayer(X, 2, 1, nn.sigmoid_tf, bias='Variable')
learn = nn.GradientDescentOptimizer(Y, output)

run = nn.RunLearnModel("AND")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})
run.Test_1(Y, output, feed_dict={X:x_data, Y:y_data})
