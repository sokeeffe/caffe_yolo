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

def grow_right(network_output_df, image_index, box_x, box_y, feature_size):
  if (box_x > feature_size - 1):
    return box_x
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  # while (box_x < feature_size - 1) and network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x + 1]['conf'] > thresh:
  while (box_x < feature_size - 1) and (network_output_df.loc[(network_output_df['cell_index'] == box_y*feature_size + box_x + 1)]['grown_0.5']==1).any():
    box_x = box_x + 1
  return box_x

def grow_down(network_output_df, image_index, box_x, box_y, feature_size):
  if (box_y > feature_size - 1):
    return box_y
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  # while (box_y < feature_size - 1) and network_output_df.iloc[image_index*feature_size*feature_size + (box_y+1)*feature_size + box_x]['grown_0.5']==1:
  while (box_y < feature_size - 1) and (network_output_df.loc[(network_output_df['cell_index'] == (box_y+1)*feature_size + box_x)]['grown_0.5']==1).any():
    box_y = box_y + 1
  return box_y

def is_cell_added(boxes, box_x, box_y):
  box_added = False
  for box in boxes:
    # print vars(box)
    # print "newline\n"
    if hasattr(box, 'xmin'):
      if box_x >= box.xmin and box_x <= box.xmin+box.width and box_y >= box.ymin and box_y <= box.ymin+box.height:
        box_added = True
  return box_added

if len(sys.argv) != 2:
  print "Usage: python extract_grown_detections.py dataframe_1.csv"
  exit()

image_dim = 288
pool_factor = 32
feature_size = image_dim/pool_factor

boxes=[Box] * (feature_size*feature_size)
output_boxes=[Box] * (feature_size*feature_size)

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
test_df1 = df1.loc[(df1['image_name']==test_image) & (df1['grown_0.5']==1)]
print test_df1

for index, box in enumerate(boxes):
  box_x = index%feature_size
  box_y = index/feature_size
  print 'index: '+str(index)+'\tbox X: '+str(box_x)+'\tbox Y: '+str(box_y)
  if is_cell_added(output_boxes, box_x, box_y):
    continue
  image_index=0
  temp_df = test_df1.loc[(test_df1['cell_index']==index)]
  if len(temp_df.index) == 1:
    print (test_df1.loc[(test_df1['cell_index'] == box_y*feature_size + box_x + 1)]['grown_0.5']==1).any()
    xmax = grow_right(test_df1, image_index, box_x, box_y, feature_size)
    ymax = grow_down(test_df1, image_index, box_x, box_y, feature_size)
    print 'xmax: '+str(xmax)+'\tymax: '+str(ymax)
    # Add box to output_boxes
    output_boxes[index] = Box(box_x, box_y, xmax - box_x + 1, ymax - box_y+1)
    print "output boxes: "
    print vars(output_boxes[index])
# Output output_boxes to a file
