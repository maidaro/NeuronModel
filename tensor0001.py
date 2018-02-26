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

layer = nn.NnFullConnectedLayer(X, 2, 1, nn.sigmoid_tf, bias='Variable')
learn = nn.GradientDescentOptimizer(Y, layer)

run = nn.RunLearnModel("OR")
run.Learn(learn, feed_dict={X:x_data, Y:y_data})

print("\n=== Test OR Implement ===")
err = 0
num_data = len(x_data)
for i in range(num_data):
    x_i = x_data[i:i +1] # shape of x_data[0] and x_data[:] (slice) is different
    y_i = y_data[i:i +1]
    out = run.Recall(layer, feed_dict={X:x_i, Y:y_i})
    #err = err + tf.square(tf.reduce_sum(y_i - out))
    print("I:{}, O:{}/R:{}".format(x_i, y_i, out))

#print("\nTotal Error : {}".format(tf.sqrt(err)))
