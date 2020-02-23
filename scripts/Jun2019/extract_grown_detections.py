'''
This script reads in a dataframe with images and corresponding detections
post processing. The script identifies the coordinates of the region proposals 
for each image and writes the proposals and corresponding labels to a 
pre configured location
Input Params:
 - dataframe_1.csv

 @author Simon
 @date 26 Jun 2019
'''
import sys
import os
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

class Box:
  def __init__(self, xmin=0, ymin=0, width=0, height=0, occ=0, neigh=0, blob=-1):
    self.xmin = xmin
    self.ymin = ymin
    self.width = width
    self.height = height
    self.occ = occ
    self.neigh = neigh
    self.blob = blob

# grow_right_one aims to expand the blob right one cell but needs at least one
# new cell to be originally detected as and object and connected on the left to
# a cell that is also detected as an object 
def grow_right_one(network_output_df, image_index, box_x, box_y, xmax, ymax, feature_size):
  # Check that box_x is not at the last cell on right
  if (box_x >= feature_size - 1):
    return box_x
  next_cells_grown = True
  connected_cell_conf = False
  true_detection = False
  # Is the current cell a true detection
  for y in range(box_y, ymax+1):
    if (network_output_df.loc[(network_output_df['cell_index'] == y*feature_size + box_x)]['conf']>0.5).any():
      true_detection = True
  # Get range of cells to grow right
  # print "True Detection: " + str(true_detection)
  if true_detection:
    for y in range(box_y, ymax+1):
      if not (network_output_df.loc[(network_output_df['cell_index'] == y*feature_size + box_x + 1)]['grown_0.5']==1).any():
        next_cells_grown = False
      if (network_output_df.loc[(network_output_df['cell_index'] == y*feature_size + box_x + 1)]['conf']>0.5).any() \
      and (network_output_df.loc[(network_output_df['cell_index'] == y*feature_size + box_x)]['conf']>0.5).any():
        connected_cell_conf = True
    if next_cells_grown and connected_cell_conf:
      box_x = box_x + 1
  else:
    for y in range(box_y, ymax+1):
      if not (network_output_df.loc[(network_output_df['cell_index'] == y*feature_size + box_x + 1)]['grown_0.5']==1).any():
        next_cells_grown = False
    if next_cells_grown:
      box_x = box_x + 1
  return box_x

## grow_down_one needs to grown down across the entire block
def grow_down_one(network_output_df, image_index, box_x, box_y, xmax, ymax, feature_size):
  # Check that box_y is not at the last cell at the bottom
  if (box_y >= feature_size - 1):
    return box_y
  next_cells_grown = True
  connected_cell_conf = False
  true_detection = False
  for x in range(box_x, xmax+1):
    if (network_output_df.loc[(network_output_df['cell_index'] == (box_y)*feature_size + x)]['conf']>0.5).any():
      true_detection = True
  # Get range of cells to grown down
  if true_detection:
    for x in range(box_x, xmax+1):
      if not (network_output_df.loc[(network_output_df['cell_index'] == (box_y+1)*feature_size + x)]['grown_0.5']==1).any():
        next_cells_grown = False
      if (network_output_df.loc[(network_output_df['cell_index'] == (box_y+1)*feature_size + x)]['conf']>0.5).any() \
      and (network_output_df.loc[(network_output_df['cell_index'] == (box_y)*feature_size + x)]['conf']>0.5).any():
        connected_cell_conf = True
    if next_cells_grown and connected_cell_conf:
      box_y = box_y + 1
  else:
    for x in range(box_x, xmax+1):
      if not (network_output_df.loc[(network_output_df['cell_index'] == (box_y+1)*feature_size + x)]['grown_0.5']==1).any():
        next_cells_grown = False
    if next_cells_grown:
      box_y = box_y + 1
  return box_y

def is_cell_added(boxes, box_x, box_y):
  box_added = False
  for box in boxes:
    # print vars(box)
    # print "newline\n"
    if box_x >= box.xmin and box_x <= box.xmin+box.width-1 and box_y >= box.ymin and box_y <= box.ymin+box.height-1:
      box_added = True
  return box_added

if len(sys.argv) != 2:
  print "Usage: python extract_grown_detections.py dataframe_1.csv"
  exit()

image_dim = 288
pool_factor = 32
feature_size = image_dim/pool_factor

boxes=[Box] * (feature_size*feature_size)
output_boxes = []

for i in range(feature_size):
  for j in range(feature_size):
    ymin = i*pool_factor
    xmin = j*pool_factor
    index = i*feature_size + j
    boxes[index] = Box(xmin, ymin, pool_factor, pool_factor)

dataframe_1_file = sys.argv[1]

df1 = pd.read_csv(dataframe_1_file)

# Get a list of images
image_files = df1.image_name.unique()

# Temp work - try get working for one image
test_image = image_files[0]
# test_df1 = df1.loc[(df1['image_name']==test_image) & (df1['grown_0.5']==1)]


for image_index, image_file in enumerate(image_files):
  # image_index = 17
  # image_file = 'oisin_compete0393'
  test_df1 = df1.loc[(df1['image_name']==image_file) & (df1['grown_0.5']==1)]
  # print test_df1
  output_boxes = []
  print "\nImage Index: " + str(image_index)
  print "Image File: " + image_file
  for index, box in enumerate(boxes):
    box_x = index%feature_size
    box_y = index/feature_size
    if is_cell_added(output_boxes, box_x, box_y):
      continue
    temp_df = test_df1.loc[(test_df1['cell_index']==index)]
    # print temp_df
    if len(temp_df.index) == 1:
      # print 'index: '+str(index)+'\tbox X: '+str(box_x)+'\tbox Y: '+str(box_y)
      # print (test_df1.loc[(test_df1['cell_index'] == box_y*feature_size + box_x + 1)]['grown_0.5']==1).any()
      xmax = box_x
      ymax = box_y
      prev_xmax = box_x
      prev_ymax = box_y
      xmax = grow_right_one(test_df1, image_index, box_x, box_y, xmax, ymax, feature_size)
      ymax = grow_down_one(test_df1, image_index, box_x, box_y, xmax, ymax, feature_size)
      while(xmax != prev_xmax or ymax != prev_ymax):
        prev_xmax = xmax
        prev_ymax = ymax
        xmax = grow_right_one(test_df1, image_index, prev_xmax, box_y, xmax, ymax, feature_size)
        ymax = grow_down_one(test_df1, image_index, box_x, prev_ymax, xmax, ymax, feature_size)
        # print 'xmax: '+str(xmax)+'\tymax: '+str(ymax)
      # Add box to output_boxes
      output_boxes.append(Box(box_x, box_y, xmax - box_x + 1, ymax - box_y+1))
      # print "output boxes: "
      # print vars(output_boxes[-1])
    # Print out the output boxes
  for box_index, box in enumerate(output_boxes):
    print "Output Box: " + str(box_index) + "\nXmin: " + str(box.xmin) + \
    " Ymin: " + str(box.ymin) + " Width: " + str(box.width) + " Height: " + \
    str(box.height)
  if image_index == 49:
    break
# Output output_boxes to a file
