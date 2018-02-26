import tensorflow as tf
import tensormodel as nn

# implement OR NN machine relu function

# define input data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# define output data
y_data = [[0], [1], [1], [1]]

X = nn.DefPlaceHolder(2, 'Input')
Y = nn.DefPlaceHolder(1, 'Target')

layer = nn.NnFullConnectedLayer(X, 2, 2, nn.relu_tf, bias='Variable')
hidden = nn.NnFullConnectedLayer(layer.activation, 2, 1, nn.relu_tf, bias='Variable')
# Relu function with GradientDescentOptimizer does not work to learn OR operator
## learn = nn.GradientDescentOptimizer(Y, layer)
# Relu function with SoftmaxCrossEntropy in Single layer does not work
## learn = nn.SoftmaxCrossEntropy(Y, layer)
# Working Relu function requires at least more than one layer.
# However result is acceptable not always because optimizer can not find answer somtimes.
learn = nn.GradientDescentOptimizer(Y, hidden)

run = nn.RunLearnModel("OR with Relu")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})

print("\n=== Test 'OR with Relu(softmax)' Implement ===")
num_data = len(x_data)
for i in range(num_data):
    x_i = x_data[i:i +1] # shape of x_data[0] and x_data[:] (slice) is different
    y_i = y_data[i:i +1]
    print("I:{}, O:{}/R:{}".format(x_i, y_i, run.Recall(hidden, feed_dict={X:x_i, Y:y_i})))
