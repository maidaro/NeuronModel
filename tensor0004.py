import tensorflow as tf
import tensormodel as nn

# implement OR NN machine relu function

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [[0], [1], [1], [1]]

X = nn.DefPlaceHolder(2, 'Input')
Y = nn.DefPlaceHolder(1, 'Target')

hidden = nn.NnFullConnectedLayer(X, 2, 2, nn.relu_tf, bias='Variable')
output = nn.NnFullConnectedLayer(hidden.activation, 2, 1, nn.relu_tf, bias='Variable')
# Relu function with GradientDescentOptimizer does not work to learn OR operator
## learn = nn.GradientDescentOptimizer(Y, layer)
# Relu function with SoftmaxCrossEntropy in Single layer does not work
## learn = nn.SoftmaxCrossEntropy(Y, layer)
# Working Relu function requires at least more than one layer.
# However result is acceptable not always because optimizer can not find answer somtimes.
learn = nn.GradientDescentOptimizer(Y, output)

run = nn.RunLearnModel("OR with Relu(softmax)")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})
run.Test_1(Y, output, feed_dict={X:x_data, Y:y_data})
