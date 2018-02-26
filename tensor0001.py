import tensorflow as tf
import numpy as np
import tensormodel as nn

# implement OR NN machine using tensorflow

# define input pattern
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# define output pattern, rank of output pattern should be equal to input, or tensorflow broadcast output tensor and result is unpredictable
y_data = np.array([[0], [1], [1], [1]])

X = tf.placeholder(tf.float32, [None, 2], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

output = nn.NnFullConnectedLayer(X, 2, 1, nn.sigmoid_tf, bias='Variable')
learn = nn.GradientDescentOptimizer(Y, output)

run = nn.RunLearnModel("OR")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})
run.Test_1(Y, output, feed_dict={X:x_data, Y:y_data})
