#!/bin/bash

# display_LMDB-image.sh
# Script takes a lmdb image input in the form of a csv file and matching
# labels also stored in a csv file. It first processes the lmdb input 
# image file to generate a jpg image with a cpp file. It calls a python 
# file which draws the labels on the image and saves the image as a new
# file.

# @author: 	Simon
# @date:	18Dec2018

scriptName=$(basename "$0")

# Use code from the coding directory 
# (This should be moved into the caffe directory later)
codeDirectory=/home/simon/Coding/CPlusPlus/ImageProcessing
pythonDirectory=/home/simon/Coding/Python

image=$1
label=$2

# Script requires two parameters
#   lmdb image file
#   labels file
if [ -z $2 ] ; then
  echo "*** ${scriptName} ERROR: called with args: $*"
  echo "usage ${scriptName} net_lmdb.csv labels.txt"
  exit 1
fi

# Converts the lmdb data into a jpg image
$codeDirectory/darknet_caffe_lmdb_image_show $1 $2

# Get the name for the jpg image
image_raw="$(sed 's/lmdb/image/g;s/csv/jpg/g' <<<$image)"

cp sized_loaded.jpg $image_raw

image_labelled="$(sed 's/input/label/g' <<<$image_raw)"

# Put labels on the jpg image
$pythonDirectory/SaveYOLOData_v2.py $image_raw $label $image_labelled

image_file=$(basename "$image_raw")

#mv $image_file $image_labelled

echo $codeDirectory
