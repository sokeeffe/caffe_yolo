'''
This script performs a sanity check on the dataset generated for the second
stage of the pipeline based off forward pass of the dataset through the first
stage of the pipeline.

The pipeline refers to the piece of work carried out between Jun2018 and Dec2019

Validation script has already been run on the train dataset and the results of 
inference are stored in a csv file.
To generate the CSV the following command was run:
python scripts/Jun2019/valid_object_network_v3.py \
~/DeepLearning/datasets/SPLObjDetectDataset/train.txt \
docker_models/region_proposal9/region_proposal9_deploy.prototxt \
docker_models/region_proposal9/region_proposal9_iter_100000.caffemodel
followed by
python scripts/Jun2019/eval_grown_region_cover_v2_0_5.py \
models_June19/test_RP9/network_perf_09Oct2019.csv \
models_June19/test_RP9/network_perf_processed_09Oct2019.csv


This script performs inference on the individual images with
python scripts/Jun2019/test_object_network_v4.py \
docker_models/region_proposal9/region_proposal9_deploy.prototxt \
docker_models/region_proposal9/region_proposal9_iter_100000.caffemodel \
~/DeepLearning/datasets/SPLObjDetectDataset/train/images/eliza_ball_01070295.png \
0.5

This script will call inference on N images through the first stage of the 
pipeline and then view the labelled dataset data on the generated images for
the second stage dataset from the first N files.

 @author Simon
 @date 09Dec2019
 Outlined the approach the script will take.
 @date 10Dec2019
 Added calls to run inference on first N files
 Added calls to visualize image with corresponding labels
 @date 11Dec2019
 Modified hardcoded paths to point to validation dataset split
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

first_stage_config = \
'/home/simon/DeepLearning/caffe/docker_models/region_proposal9/region_proposal9_deploy.prototxt'
first_stage_weights = \
'/home/simon/DeepLearning/caffe/docker_models/region_proposal9/region_proposal9_iter_100000.caffemodel'
image_list = '/home/simon/DeepLearning/datasets/SPLObjDetectDataset/val.txt'
threshold = '0.5'

f = open(image_list,'r')
images = f.readlines()
f.close()

for idx, image in enumerate(images):
  # Remove the new line at the end of the file path
  image_file = image.split('\n')[0]
  label_file = image_file.replace("images","labels")
  label_file = label_file.replace("png","txt")
  # This script must be run from /home/simon/DeepLearning/caffe directory
  system_call = 'python scripts/Jun2019/test_object_network_v4.py ' + \
  first_stage_config + ' ' + first_stage_weights + ' ' + image_file + ' ' + \
  threshold
  os.system(system_call)
  image_basename = os.path.basename(image_file)
  image_path = '/home/simon/DeepLearning/caffe/temp_output/'+image_basename
  system_call = 'python /home/simon/Coding/Python/SaveYOLOData_v2.py ' + \
  image_path + ' ' + label_file
  os.system(system_call)
  if idx >= 9:
    break

second_stage_dir = '/home/simon/DeepLearning/datasets/RP9/val/images/'
onlyfiles = [f for f in listdir(second_stage_dir) if isfile(join(second_stage_dir, f))]

for idx, image in enumerate(onlyfiles):
  image_file = second_stage_dir+image
  label_file = image_file.replace("images","labels")
  label_file = label_file.replace("png","txt")
  system_call = 'python /home/simon/Coding/Python/SaveYOLOData_v2.py ' + \
  image_file + ' ' + label_file
  os.system(system_call)

