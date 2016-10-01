## cnn_trainer.py
##
## This script trains a CNN to perform classification on single chest sensor data

import tensorflow as tf
import numpy as np

from datasets import SingleChest

#from models.CNN.testModel1 import hidden_layers
from models.CNN import *

import batcher

import random

BATCH_SIZE = 250
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-2

# METADATA FOR WINDOW SIZE
WINDOW_SIZE = 26

tgt_prob = [0, 0.278, 1.0, 0.835, 0.348, 1.0, 1.0, 0.112]

# Functions to load and process data

def create_batcher(dataset, in_shape=(WINDOW_SIZE, 1, 3), target_shape=(7,)):
   """
   Create a batch from the provided items
   """

   data_shape = {'input': in_shape, 'target': target_shape}
   batch = batcher.Batch(data_shape)

   # Add all the samples from the dataset list
   num_captures = dataset.get_number_captures()

   for i in range(num_captures):
      data, target = dataset.get_capture(i)
      seq_len = data.shape[0]

      if seq_len == WINDOW_SIZE:
         idx = np.argmax(target)
         if random.random() < tgt_prob[idx]:
            batch.add_sample({'input': np.reshape(data, in_shape), 'target': target[1:]})

   return batch

 
HIDDEN_LAYERS = [(CONV, 'conv1', (5, 1, 3, 20), (1,1,1,1), 'VALID'),
                 (POOL, 'pool1', (1,2,1,1), (1,2,1,1), 'VALID'),
                 (RELU, 'relu1'),
                 (CONV, 'conv2', (5, 1, 20, 30), (1,1,1,1), 'VALID'),
                 (POOL, 'pool2', (1,2,1,1), (1,2,1,1), 'VALID'),
                 (RELU, 'relu2'),
                 (FLAT, 'flat'),
                 (FULL, 'full1', 40),
                 (RELU, 'relu4')]


if __name__ == '__main__':
   """
   """



   # Load the datasets and create a batch generator

   print "Loading Dataset...",
   dataset = SingleChest.Dataset(1, WINDOW_SIZE)
   dataset.load()
   print "Done"

   batch = create_batcher(dataset)
   print "Number of samples:", batch.num_samples()

   # Setup the CNN model and tensorflow
   print "Creating Model"

   model = CNN((WINDOW_SIZE, 1, 3), HIDDEN_LAYERS, 7)       # Make a model
   optimizer = tf.train.AdamOptimizer(LEARNING_RATE)     # And something to train it with
   train_step = optimizer.minimize(model.objective())

   # With the graph build, start a session and train!
   sess = tf.InteractiveSession()                    

   sess.run(tf.initialize_all_variables())

   # Train for however many epochs
   for i in range(NUM_EPOCHS):   
      model.train_batch(train_step, batch)

      print "Epoch %d - Objective: %4.2f; Accuracy: %4.2f %%" % (i, model.get_cost(batch), 100*model.get_accuracy(batch))

      batch.shuffle()


