import tensorflow as tf
import numpy as np

def sigmoid_tf(x):
    return 1.0/(1.0 + tf.exp(-x))

def relu_tf(x):
    return tf.maximum(x, 0.0)

# define input pattern
def DefPlaceHolder(num_pattern, name = None):
    return tf.placeholder(tf.float32, [None, num_pattern], name=name)

### define NN function model
class NnFullConnectedLayer:

    def __init__(self, Input, num_in_ptn, num_out_ptn, act_func, wtf = -1.0, wtt = 1.0, bias = 'Constant', btf = -1.0, btt = 1.0):
        assert bias == 'Constant' or bias == 'Variable'

        ## W <- weight of connection
        ## b <- bias of neuron
        ## It is important to define initial value for Neuron network to find solution
        self.W = tf.Variable(tf.random_uniform([num_in_ptn, num_out_ptn], wtf, wtt), name='Weight')
        ## bias is disabled until general delta rule is used
        if bias == 'Constant':
            self.b = tf.constant(btf, dtype=tf.float32, shape=[num_out_ptn, ], name='Bias')
        elif bias == 'Variable':
            self.b = tf.Variable(tf.random_uniform([num_out_ptn, ], btf, btt))
        ## X is active value of input neuron
        ## W is weight of connection from that neuron
        # activation = tf.reduce_sum(tf.multiply(X, W), 1) + b
        ## activation is active value from neuron
        self.activation = act_func(tf.add(tf.matmul(Input, self.W), self.b))

class GradientDescentOptimizer:
    ## at the first version, I consider only recall process and missed learning process, chaning weight of NN
    ## GradientDescentOptimizer is learning model
    ## learning_rate <- Hebbian rule
    def __init__(self, Target, Output, hebbian_learning_rate = 0.1):
        self.cost_model = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(Target, Output.activation))))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=hebbian_learning_rate)
        self.train_op = optimizer.minimize(self.cost_model)

class SoftmaxCrossEntropy:
    def __init__(self, Target, Output, hebbian_learning_rate = 0.1):
        self.cost_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Target, logits=Output.activation))
        optimizer = tf.train.AdamOptimizer(learning_rate=hebbian_learning_rate)
        self.train_op = optimizer.minimize(self.cost_model)

class RunLearnModel:
    def __init__(self, name):
        self.name = name
        self.sess = tf.Session()
        ## initialize neuron network
        self.sess.run(tf.global_variables_initializer())

    def Learn(self, optimizer, feed_dict, output = 1000, epoch = 10000):

        # set learning iteration parameter
        #   - iteration count
        #   - error rate
        #   - based on normal distribution?
        for step in range(epoch):
            _, cost_val = self.sess.run([optimizer.train_op, optimizer.cost_model], feed_dict=feed_dict)
            if (step + 1) % output == 0:
                print(step + 1, cost_val)

    def Recall(self, model, feed_dict):
        return self.sess.run(model.activation, feed_dict=feed_dict)
