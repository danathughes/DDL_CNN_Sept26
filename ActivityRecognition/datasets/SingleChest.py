## SingleChest.py
##
## The SingleChest dataset consists of captures of sequences of a single accelerometer value.
##
## Creates a dataset object which loads data from the Activity Recognition from Single Chest-Mounted
## Accelerometer Data Set.  The dataset is available from the UCI Machine Learning Repository:
##
## https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer
##
## Individual entries consist of the following:
##   Step - Time step number, ignored
##   Accelerometer x,y,z
##   Class label number, 0 - 7
## 


import numpy as np

# PATHS AND FILENAMES

DATASET_PATH = "./datasets/SingleChest/"
MAX_SENSOR_VALUE = 4096.0      # This is a guess...Not sure what the sensor range is

class Dataset:
   """
   An object holding data from an opportunity dataset
   """

   def __init__(self, subject_numbers, window_size):
      """
      Create an object containing the desired subject number, which subjects to load,
      and how many sensor values to put in a window.

      subject_numbers - a list of subject numbers to load, should be in the range of
                        [1, 15]
      window_size     - how many samples should be present in each measurement window?
      """

      # Hold on to passed in values
      self._filenames = [str(s) + '.csv' for s in subject_numbers]
      self._window_size = window_size

      # Captures are windows of the provided size
      self._captures = []


   def parse_line(self, line):
      """
      Parse a line read from the input file
      """

      data = line.strip().split(',')

      x = float(data[1])
      y = float(data[2])
      z = float(data[3])

      label = float(data[4])

      return x,y,z,label


   def load(self, normalize_data = True):
      """
      Load and process the data
      """

      # Load the dataset from the files
      for filename in self._filenames:
         path = DATASET_PATH + filename
         self.load_file(path)

      # Convert the data and create targets
      self.data_to_array()


   def load_file(self, path):
      """
      Actually load the information from the appropriate file
      """

      # Read in the file

      with open(path) as datafile:

         # Read in all the data 
         dataline = datafile.readline()
         x, y, z, label = self.parse_line(dataline)

         data = [[x,y,z]]
         target = [label]

         dataline = datafile.readline()

         while dataline:

            x, y, z, label = self.parse_line(dataline)

            # Is this a new capture?
            if len(data) == self._window_size:
               # Save the prior capture and start a new one
               self._captures.append( (data,target) )
               data = [[x,y,z]]
               target = [label]
            else:
               data.append([x,y,z])
               target.append(label)

            dataline = datafile.readline()

      # Done loading in the data - store the last capture
      self._captures.append( (data, target) )


   def data_to_array(self):
      """
      Convert the list of data to a numpy array
      """

      for i in range(len(self._captures)):
         data = np.array(self._captures[i][0]) / MAX_SENSOR_VALUE
         targ = self._captures[i][1]
         tmp = np.zeros((8,))
         for t in targ:
            tmp[int(t)] += 1
         target_idx = np.argmax(tmp)
         target = np.zeros((8,))
         target[target_idx] = 1.0

         self._captures[i] = (data, target)
               

   def get_number_captures(self):
      """
      How many captures are in the dataset?
      """

      return len(self._captures)


   def get_capture(self, capture_number):
      """
      Get the capture's sensor values and target 
      """

      x = self._captures[capture_number][0]
      y = self._captures[capture_number][1]

      return x, y

