## CNN.py
##
## A simple Convolution Neural Network model.

import tensorflow as tf
import numpy as np

BATCH_SIZE = 100

CONV = 'CONVOLUTIONAL'
POOL = 'POOLING'
RELU = 'RELU'
FULL = 'FULL'
FLAT = 'FLAT'
DROPOUT = 'DROPOUT'

# Function to help build the network

def weight_variable(shape, name):
   "Creates a weight matrix of the desired shape, initialized using a Gaussian"
   return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def bias_variable(shape, name):
   "Creates a set of bias variables, initialized to 0.1"
   return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def cross_entropy(prediction, target):
   "Creates a tensor calculating the cross-entropy between the prediction and target tensor"
   return tf.reduce_mean(-tf.reduce_sum(target * tf.log(prediction + 1e-10), reduction_indices=[1]))


def create_layer(input_layer, layer):
   """
   Build one of the following types of layers
         CONV - (type, name, kernel_shape, stride, padding)
         POOL - (type, name, pool_shape, stride, padding)
         RELU - (type, name)
         FLAT - (type, name)
         FULL - (type, name, output_size)
   """

   # Common to all layers
   layer_type = layer[0]
   name = layer[1]


   if layer_type == CONV:
      weights = weight_variable(layer[2], name+'_conv_weights')
      stride = layer[3]
      padding = layer[4]
      return tf.nn.conv2d(input_layer, weights, stride, padding, name=name+'_conv')
            

   elif layer_type == POOL:
      pool_shape = layer[2]
      stride = layer[3]
      padding = layer[4]
      return tf.nn.max_pool(input_layer, pool_shape, stride, padding, name=name+'_pool')


   elif layer_type == RELU:
      # What is the shape of the activiation function?
      activation_size = input_layer.get_shape()[-1].value

      # Make a bias for the activation
      bias = bias_variable([activation_size], name+'_bias')
      return tf.nn.relu(input_layer + bias, name=name+'_relu')


   elif layer_type == FULL:
      # What's the shape of the previous layer?
      input_size = input_layer.get_shape()[-1].value
      output_size = layer[2]

      weights = weight_variable([input_size, output_size], name+'_weights')
      return tf.matmul(input_layer, weights)
            

   elif layer_type == FLAT:
      # Simply flatten the previous layer
      input_layer_shape = input_layer.get_shape()[1:].dims
      flat_dim = reduce(lambda x,y: x*y, input_layer_shape, tf.Dimension(1))

      return tf.reshape(input_layer, [-1, flat_dim.value])   


class CNN(object):

   def __init__(self, inputFrameShape, hidden_layers, num_labels, **kwargs):
      """
      Build a deep convolutional neural network network

      window_size   - the number of samples per window
      num_sensors   - the number of sensor measurements per sample
      num_labels    - the number of unique classes to identify
      hidden_layers - A list of hidden layers, which take the form of a tuple, which depends
                      on the type (first element of the tuple)
      """

      # Unpack the input frame
      num_time_steps, one, num_sensors = inputFrameShape

      # Input and target placeholders
      self._input = tf.placeholder(tf.float32, shape=[None, num_time_steps, 1, num_sensors])
      self._target = tf.placeholder(tf.float32, shape=[None, num_labels])

      # Build up the hidden layers for the network
      current_layer = self._input

      for layer in hidden_layers:
         current_layer = create_layer(current_layer, layer)

      # Create the output layer by creating a fully connected softmax layer
      input_size = current_layer.get_shape()[-1].value

      W_output = weight_variable([input_size, num_labels], 'classifier_weights')
      b_output = bias_variable([num_labels], 'classifier_bias')

      self._output = tf.nn.softmax(tf.matmul(current_layer, W_output) + b_output)

      # Set the objective to the cross entropy of the output and target
      self._objective = cross_entropy(self._output, self._target)

      # And also be able to predict the accuracy
      correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._output, 1))
      self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


   def objective(self):
      """
      Return the objective tensor of this network
      """

      return self._objective



   def accuracy(self):
      """
      Return the accuracy tensor of this network
      """

      return self._accuracy


   def train(self, train_step, x, y):
      """
      Train on the input data (x) and the target (y).  The train step is some optimizer
      """

      train_step.run(feed_dict={self._input: x, self._target: y})


   def train_batch(self, train_step, data_set, batch_size = BATCH_SIZE):
      """
      Train on the a full dataset
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      _data = data_set.get_batch(batch_size)

      while _data["input"].shape[0] > 0:
         _x = _data["input"]
         _y = _data["target"]

         self.train(train_step, _x, _y)

         _data = data_set.get_batch(batch_size)


   def get_accuracy(self, data_set, batch_size = BATCH_SIZE):
      """
      Determine the accuracy of the provided data set
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      total_correct = 0.0
      total_count = 0.0

      _data = data_set.get_batch(batch_size)

      while _data["input"].shape[0] > 0:
         _x = _data["input"]
         _y = _data["target"]

         total_count += _x.shape[0]
         acc = self._accuracy.eval(feed_dict = {self._input: _x, self._target: _y})
         total_correct += acc * _x.shape[0]

         _data = data_set.get_batch(batch_size)

      data_set.set_index(index_snapshot)

      return total_correct / total_count


   def get_cost(self, data_set, batch_size = BATCH_SIZE):
      """
      Determine the cost of the provided data set
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      total_cost = 0.0

      _data = data_set.get_batch(batch_size)

      while _data["input"].shape[0] > 0:
         _x = _data["input"]
         _y = _data["target"]

         cost = self._objective.eval(feed_dict = {self._input: _x, self._target: _y})
         total_cost += cost

         _data = data_set.get_batch(batch_size)

      data_set.set_index(index_snapshot)

      return total_cost / data_set.num_samples()

