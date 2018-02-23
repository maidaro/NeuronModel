import tensorflow as tf
import numpy as np

def sigmoid_tf(x):
    return 1.0/(1.0 + tf.exp(-x))

# implement OR NN machine using tensorflow

# define input pattern
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# define target pattern
y_data = np.array([0, 1, 1, 1])

### define NN function model
class NnModelCell:

    X = tf.placeholder(tf.float32, name="X")

    def __init__(self, i_rank, t_rank, act_func):
        ## W <- weight of connection
        ## b <- bias of neuron
        ## It is important to define initial value for Neuron network to find solution
        self.W = tf.Variable(tf.random_uniform([i_rank, ], -1.0, 1.0), name='Weight')
        ## bias is disabled until general delta rule is used
        # b = tf.Variable(tf.zeros([1, ]))
        self.b = tf.constant(0, dtype=tf.float32, shape=[t_rank, ], name='Bias')
        # hypothesis = tf.reduce_sum(tf.multiply(X, W), 1) + b
        ## hypothesis is active value from neuron
        ## X is active value of input neuron
        ## W is weight of connection from that neuron

        self.hypothesis = act_func(tf.reduce_sum(tf.multiply(self.X, self.W), i_rank - 1) + self.b)

# define optimization model
class LearningModel:
    Y = tf.placeholder(tf.float32, name="Y")

    ## at the first version, I consider only recall process and missed learning process, chaning weight of NN
    ## learning_rate <- Hebbian rule
    def __init__(self, t_rank, hebbian_learning_rate, model):
        cost = tf.sqrt(tf.reduce_sum(tf.square(model.hypothesis - self.Y), 0))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(cost)

def variable_summaries(var, name = None):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_' + name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
