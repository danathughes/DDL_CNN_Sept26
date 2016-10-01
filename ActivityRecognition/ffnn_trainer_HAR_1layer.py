## cnn_trainer_HAR_1layer.py
##
## This script trains a CNN to perform classification on single chest sensor data.
## A single CNN layer is used, followed by a fully connected layer

import tensorflow as tf
import numpy as np

from datasets import SingleChest
from models.CNN import *
import batch_generator

import random

# Learning parameters
BATCH_SIZE = 250
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3

# How to handle the provided dataset
SUBJECT_NUMBERS = range(1,16)      # All the data
WINDOW_SIZE = 26
DATA_SPLIT = (0.7, 0.15, 0.15)     # Percent split for training, validation and test data

# Since the number of samples for each class are not evenly distributed, randomly
# keep each sample so that the number of samples for each class is roughly the 
# same.  Note that this is a bit of a hack, the more appropriate approach would be
# to weight the cost function based on prior probability.
tgt_prob = [0.08, 1.0, 0.22, 0.13, 0.93, 1.0, 0.08]


def create_batcher(dataset, in_shape=(WINDOW_SIZE, 1, 3), target_shape=(7,)):
   """
   Create a batch generator from the desired dataset.
   """

   data_shape = {'input': in_shape, 'target': target_shape}
   batch = batch_generator.BatchGenerator(data_shape)

   # Add all the samples from the dataset list
   num_captures = dataset.get_number_captures()

   for i in range(num_captures):
      data, target = dataset.get_capture(i)
      seq_len = data.shape[0]

      target = target[1:]   # Class '0' is never used...

      if seq_len == WINDOW_SIZE:   # Ignore any incomplete windows from end of the dataset
         idx = np.argmax(target)
         if random.random() < tgt_prob[idx]:
            batch.add_sample({'input': np.reshape(data, in_shape), 'target': target})

   return batch

# 6807 Parameters
HIDDEN_LAYERS = [(FLAT, 'flat'),
                 (FULL, 'full1', 200),
                 (RELU, 'relu1')]


def run():
   """
   Load the data, train a CNN and output the results of the training
   """

   # Load the datasets and create a batch generator

   print "Loading Dataset...",
   dataset = SingleChest.Dataset(SUBJECT_NUMBERS, WINDOW_SIZE)
   dataset.load()
   print "Done"

   # Create a training, validation and test set
   batch_gen = create_batcher(dataset)
   batch_gen.shuffle()
   train_set, validation_set, test_set = batch_gen.split(DATA_SPLIT)

   print "Number of samples:"
   print "  Training Set:  ", train_set.num_samples()
   print "  Validation Set:", validation_set.num_samples()
   print "  Test Set:      ", test_set.num_samples()

   # Setup the CNN model and tensorflow.  There are 3 accelerometer values, and 7 target labels
   print "Creating Model"

   model = CNN((WINDOW_SIZE, 1, 3), HIDDEN_LAYERS, 7)       # Make a model
   optimizer = tf.train.AdamOptimizer(LEARNING_RATE)        # And something to train it with
   train_step = optimizer.minimize(model.objective())

   # With the graph build, start a session and train!
   sess = tf.InteractiveSession()                    
   sess.run(tf.initialize_all_variables())

   # Pre-training stats
   train_cost = model.get_cost(train_set)
   train_accuracy = 100*model.get_accuracy(train_set)
   validation_cost = model.get_cost(validation_set)
   validation_accuracy = 100*model.get_accuracy(validation_set)

   print "Epoch %d - Training Objective: %4.6f; Training Accuracy: %4.2f %%\tValidation Objective: %4.6f; Validation Accuracy: %4.2f %%" % (0, train_cost, train_accuracy, validation_cost, validation_accuracy)

   # Train for however many epochs
   for i in range(NUM_EPOCHS):   
      model.train_batch(train_step, train_set)

      train_cost = model.get_cost(train_set)
      train_accuracy = 100*model.get_accuracy(train_set)
      validation_cost = model.get_cost(validation_set)
      validation_accuracy = 100*model.get_accuracy(validation_set)

      print "Epoch %d - Training Objective: %4.6f; Training Accuracy: %4.2f %%\tValidation Objective: %4.6f; Validation Accuracy: %4.2f %%" % (i+1, train_cost, train_accuracy, validation_cost, validation_accuracy)

      # Shuffle the batch
      train_set.shuffle()

   # All done!
   train_cost = model.get_cost(train_set)
   train_accuracy = 100*model.get_accuracy(train_set)
   validation_cost = model.get_cost(validation_set)
   validation_accuracy = 100*model.get_accuracy(validation_set)
   test_cost = model.get_cost(test_set)
   test_accuracy = 100*model.get_accuracy(test_set)

   print "Final Results:"
   print "  Training Objective: %4.6f; Training Accuracy: %4.2f %%" % (train_cost, train_accuracy)
   print "  Validation Objective: %4.6f; Validation Accuracy: %4.2f %%" % (validation_cost, validation_accuracy)
   print "  Test Objective: %4.6f; Test Accuracy: %4.2f %%" % (test_cost, test_accuracy)


if __name__== "__main__":
   run()

