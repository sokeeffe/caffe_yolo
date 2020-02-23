'''
This script generates the train.txt, val.txt, and test.txt files necessary
to create the LMDB dataset format requried for Caffe. The txt files are simply 
a list of all images in the dataset

Although Caffe randomizes the dataset order when creating the LMDB format if 
the option is selected, the list of images for the train, val, and test text 
files will also be randomized in case the randomized option is ever not
selected when creating the LMDB format.

 @author Simon
 @date 11Dec2019
 Initial revision of script
'''
import sys
import os
from os import listdir
from os.path import isfile, join
import Image
import math
import glob
import random
import numpy as np
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Train dataset split
second_stage_train_dir = '/home/simon/DeepLearning/datasets/RP9/train/images/'
onlyfiles_train = [f for f in listdir(second_stage_train_dir) if isfile(join(second_stage_train_dir, f))]
# Add file path to each image file
onlyfiles_path_train = [second_stage_train_dir + s for s in onlyfiles_train]
# Randomize list
random.shuffle(onlyfiles_path_train)
output_file_train = '/home/simon/DeepLearning/datasets/RP9/train.txt'
with open(output_file_train, 'w') as filehandle:
  filehandle.writelines("%s\n" % item for item in onlyfiles_path_train)

# Val dataset split
second_stage_val_dir = '/home/simon/DeepLearning/datasets/RP9/val/images/'
onlyfiles_val = [f for f in listdir(second_stage_val_dir) if isfile(join(second_stage_val_dir, f))]
# Add file path to each image file
onlyfiles_path_val = [second_stage_val_dir + s for s in onlyfiles_val]
# Randomize list
random.shuffle(onlyfiles_path_val)
output_file_val = '/home/simon/DeepLearning/datasets/RP9/val.txt'
with open(output_file_val, 'w') as filehandle:
  filehandle.writelines("%s\n" % item for item in onlyfiles_path_val)

# Test dataset split
second_stage_test_dir = '/home/simon/DeepLearning/datasets/RP9/test/images/'
onlyfiles_test = [f for f in listdir(second_stage_test_dir) if isfile(join(second_stage_test_dir, f))]
# Add file path to each image file
onlyfiles_path_test = [second_stage_test_dir + s for s in onlyfiles_test]
# Randomize list
random.shuffle(onlyfiles_path_test)
output_file_test = '/home/simon/DeepLearning/datasets/RP9/test.txt'
with open(output_file_test, 'w') as filehandle:
  filehandle.writelines("%s\n" % item for item in onlyfiles_path_test)