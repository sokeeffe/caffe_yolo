#!/bin/bash

# visualize_inputs.sh
# Script takes a series lmdb image inputs in the form of csv files and matching
# labels also stored in a csv file. It first processes the lmdb input 
# image file to generate a jpg image with a cpp file. It calls a python 
# file which draws the labels on the image and saves the image as a new
# file.

# @author:  Simon
# @date:  21Feb2020

scriptName=$(basename "$0")

# Use code from the coding directory 
# (This should be moved into the caffe directory later)
codeDirectory=/home/simon/Coding/CPlusPlus/ImageProcessing
pythonDirectory=/home/simon/Coding/Python

for batch in {0..498}
do
  # Converts the lmdb data into a jpg image
  $codeDirectory/darknet_caffe_lmdb_image_show_var models_May19/region_proposal9/lmdb_input_"$batch"_0_3_64_64.csv models_May19/region_proposal9/modified_labels_"$batch"_0_3_64_64.csv 64
  cp sized_loaded.jpg models_May19/region_proposal9/image_input_"$batch"_0_3_64_64.jpg
done

for batch in {0..498}
do
  $pythonDirectory/SaveYOLOData_v2.py models_May19/region_proposal9/image_input_"$batch"_0_3_64_64.jpg models_May19/region_proposal9/modified_labels_"$batch"_0_3_64_64.csv
done


