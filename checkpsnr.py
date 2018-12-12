from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_LaMemDataset as lamem
# import read_FlowersDataset as flowers
import datetime
import BatchDatsetReader as dataset
from   six.moves import xrange
import os
import glob
import math


file_list=[]
file_glob = os.path.join('/home/shikhar/Desktop/full run/lamem', "images", '*.' + 'jpg')
file_list.extend(glob.glob(file_glob))
train_images=file_list

gen_loss_mse = 20 * math.log10(255.0 / math.sqrt(np.mean( train_images[0] - train_images[1]) ** 2 ))
print(gen_loss_mse)