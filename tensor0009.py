import tensorflow as tf
import tensormodel as nn

# XOR with Relu (delta)
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = nn.DefPlaceHolder(2, 'Input')
Y = nn.DefPlaceHolder(1, 'Target')

hidden = nn.NnFullConnectedLayer(X, 2, 2, nn.relu_tf, bias='Variable')
output = nn.NnFullConnectedLayer(hidden.activation, 2, 1, nn.relu_tf, bias='Variable')
learn = nn.GradientDescentOptimizer(Y, output)

run = nn.RunLearnModel("XOR with Relu")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})
run.Test_1(Y, output, feed_dict={X:x_data, Y:y_data})
