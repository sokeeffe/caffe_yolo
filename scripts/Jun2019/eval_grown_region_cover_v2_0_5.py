'''
This script analyses the SPLObjDetect dataset and predicts the amount of
image covered with a possible grid based region proposal
It considers the image coverage for different feature map spatial dimensions
It required that the connected grids grouped in rectangular blobs
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
  def __init__(self, xmin, ymin, width, height, occ=0, neigh=0, blob=-1):
    self.xmin = xmin
    self.ymin = ymin
    self.width = width
    self.height = height
    self.occ = occ
    self.neigh = neigh
    self.blob = blob

class Blob:
  def __init__(self, xmin, ymin, width, height):
    self.xmin = xmin
    self.ymin = ymin
    self.width = width
    self.height = height

def overlap(x1, w1, x2, w2): #x1 ,x2 are two box xmin
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    return right - left

def box_intersect(box, truth):
    w = overlap(box.xmin, box.width, truth.xmin, truth.width)
    h = overlap(box.ymin, box.height, truth.ymin, truth.height)
    if w < 0 or h < 0:
      return 0
    else:
      return 1

def blob_intersect(box, blobs):
  if not blobs:
    return False
  for blob in blobs:
    w = overlap(box.xmin, box.width, blob.xmin, blob.width)
    h = overlap(box.ymin, box.height, blob.ymin, blob.height)
    if w > 0 and h > 0:
      return 1
  return 0

def search_up(boxes, min_row, min_col, max_col, side):
  for c in range(min_col, max_col+1):
    if boxes[min_row*side+c].occ:
      for r in range(min_row-1,-1,-1):
        if boxes[r*side+c].occ:
          min_row = min(min_row,r)
        else:
          break 
  return min_row

def search_down(boxes, max_row, min_col, max_col, side):
  for c in range(min_col, max_col+1):
    if boxes[max_row*side+c].occ:
      for r in range(max_row+1, side):
        if boxes[r*side+c].occ:
          max_row = max(max_row,r)
        else:
          break
  return max_row

def search_left(boxes, min_col, min_row, max_row, side):
  for r in range(min_row, max_row+1):
    if boxes[r*side+min_col].occ:
      for c in range(min_col-1,-1,-1):
        if boxes[r*side+c].occ:
          min_col = min(min_col,c)
        else:
          break
  return min_col

def search_right(boxes, max_col, min_row, max_row, side):
  for r in range(min_row, max_row):
    if boxes[r*side+max_col].occ:
      for c in range(max_col+1,side):
        if boxes[r*side+c].occ:
          max_col = max(max_col,c)
        else:
          break
  return max_col

def grow_box(boxes, row, col, side):
  print "\tRow: " + str(row) + " Col: " + str(col)
  box = boxes[row*side+col]
  xmin=box.xmin
  ymin=box.ymin
  xmax=box.xmin+box.width
  ymax=box.ymin+box.height
  min_row = row
  min_col = col
  max_row = row
  max_col = col
  blob_grown = False
  blob_grown_up = False
  blob_grown_down = False
  blob_grown_left = False
  blob_grown_right = False
  while not blob_grown:
    up = search_up(boxes, min_row, min_col, max_col, side)
    if up != min_row:
      min_row=up
      blob_grown_up=False
    else:
      blob_grown_up=True
    down = search_down(boxes, max_row, min_col, max_col, side)
    if down != max_row:
      max_row=down
      blob_grown_down=False
    else:
      blob_grown_down=True
    left = search_left(boxes, min_col, min_row, max_row, side)
    if left != min_col:
      min_col=left
      blob_grown_left = False
    else:
      blob_grown_left = True
    right = search_right(boxes, max_col, min_row, max_row, side)
    if right != max_col:
      max_col = right
      blob_grown_right = False
    else:
      blob_grown_right = True
    if blob_grown_up and blob_grown_down and blob_grown_left and blob_grown_right:
      blob_grown = True
  xmin = min_col*box.width
  ymin = min_row*box.height
  xmax = max_col*box.width+box.width
  ymax = max_row*box.height+box.height
  print "Xmin: " + str(xmin) + " YMin: " + str(ymin) + " XMax: " + str(xmax) +" YMax: " + str(ymax)
  return Blob(xmin, ymin, xmax-xmin, ymax-ymin)

def grow_left(network_output_df, image_index, box_x, box_y, thresh, feature_size):
  if (box_x < 0):
    return box_x
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  while (box_x > 0) and network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1]['conf'] > thresh:
    box_x = box_x - 1
  return box_x

def grow_up(network_output_df, image_index, box_x, box_y, thresh, feature_size):
  if (box_y < 0):
    return box_y
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  while (box_y > 0) and network_output_df.iloc[image_index*feature_size*feature_size + (box_y-1)*feature_size + box_x]['conf'] > thresh:
    box_y = box_y - 1
  return box_y

def grow_right(network_output_df, image_index, box_x, box_y, thresh, feature_size):
  if (box_x > feature_size - 1):
    return box_x
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  while (box_x < feature_size - 1) and network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x + 1]['conf'] > thresh:
    box_x = box_x + 1
  return box_x

def grow_down(network_output_df, image_index, box_x, box_y, thresh, feature_size):
  if (box_y > feature_size - 1):
    return box_y
  # print network_output_df.iloc[image_index*feature_size*feature_size + box_y*feature_size + box_x - 1 - 1]['conf']
  while (box_y < feature_size - 1) and network_output_df.iloc[image_index*feature_size*feature_size + (box_y+1)*feature_size + box_x]['conf'] > thresh:
    box_y = box_y + 1
  return box_y

if len(sys.argv) < 3:
  print "Usage python eval_region_cover_v2.py network_perf_processed.csv output.csv"
  exit()

image_dim = 288
pool_factor = 32
if image_dim%pool_factor != 0:
  print "pool_factor must be a factor of image_dim"
  exit()
feature_size = image_dim/pool_factor

boxes=[Box] * (feature_size*feature_size)

for i in range(feature_size):
  for j in range(feature_size):
    ymin = i*pool_factor
    xmin = j*pool_factor
    index = i*feature_size + j
    boxes[index] = Box(xmin, ymin, pool_factor, pool_factor)

occ_areas=[]

network_perf_processed_file = sys.argv[1]
network_perf_processed_df = pd.read_csv(network_perf_processed_file)

network_output_df = network_perf_processed_df

image_files = network_perf_processed_df.image_name.unique()

# image_file = 'robby_real_bottom0678'

threshold_range = np.arange(0.1,1.0,0.1)
# for idx, thresh in enumerate(threshold_range):
thresh = 0.5
grown_string = 'grown_0.5'
network_output_df[grown_string] = 0
for image_index, image_file in enumerate(image_files):
  for index, box in enumerate(boxes):
    box_x = index%feature_size
    box_y = index/feature_size
    
    # print network_output_df.loc[(network_output_df['image_name']==image_file) & (network_output_df['cell_index']==index), 'conf']
    temp_df = network_output_df.loc[(network_output_df['image_name']==image_file) & (network_output_df['cell_index']==index) & (network_output_df['conf'] > thresh)]
    if len(temp_df.index) == 1:
      # image_index = 0
      xmin = grow_left(network_output_df, image_index, box_x, box_y, thresh, feature_size)
      ymin = grow_up(network_output_df, image_index, box_x, box_y, thresh, feature_size)
      xmax = grow_right(network_output_df, image_index, box_x, box_y, thresh, feature_size)
      ymax = grow_down(network_output_df, image_index, box_x, box_y, thresh, feature_size)
      for xval in range(xmin, xmax+1):
        for yval in range(ymin,ymax+1):
          # network_output_df.iloc[image_index*feature_size*feature_size + yval*feature_size + xval][grown_string] = 1
          network_output_df[grown_string][image_index*feature_size*feature_size + yval*feature_size + xval] = 1
      # print str(thresh) + '\n'
      # print temp_df['conf']
      # print str(xmin) + ',' + str(box_x) + ',' + str(xmax) + ',   ' + str(ymin) + ',' + str(box_y) + ',' + str(ymax)
      # print '\n'



network_output_df.to_csv(sys.argv[2])


# columns = ['image_name', 'num_objects','area_covered','grown_area_covered']
# imageDataDf = pd.DataFrame(index=np.arange(len(image_files)),columns=columns)

# blob_columns = ['image_name', 'xmin', 'ymin', 'width','height','area']
# blobDataDf = pd.DataFrame(columns=blob_columns)

# for idx, line in enumerate(image_files):
#   image_file = line.split('\n')[0]
#   # image_file = "/home/simon/DeepLearning/datasets/SPLObjDetectDataset/train/images/pip_ball_bottom_robots0270.png"
#   basename = os.path.basename(image_file).split('.')[0]
#   label_file = image_file.replace("images","labels")
#   label_file = label_file.replace("png","txt")

#   for box in boxes:
#     box.occ = 0

#   print "\n\nbasename: " + basename

#   imageDataDf.loc[idx,'image_name'] = basename

#   if idx%200 == 0:
#     im = np.array(Image.open(image_file).resize((288,288), Image.ANTIALIAS))
#     fig, ax = plt.subplots(1)
#     ax.imshow(im)

#   blobs = []

#   gt_boxes = []
#   f = open(label_file,'r')
#   lines = f.readlines()
#   f.close()
#   for line in lines:
#     box_dims = line.rstrip('\n').split(' ')
#     xmin = int((float(box_dims[1])-float(box_dims[3])/2.0)*image_dim)
#     ymin = int((float(box_dims[2])-float(box_dims[4])/2.0)*image_dim)
#     width = int(float(box_dims[3])*image_dim)
#     height = int(float(box_dims[4])*image_dim)
#     gt_boxes.append(Box(xmin,ymin,width,height))

#   imageDataDf.loc[idx,'num_objects'] = len(lines)

#   box_area = pool_factor*pool_factor
#   occ_area = 0
#   neigh_area = 0
#   total_area = image_dim*image_dim

#   for index, box in enumerate(boxes):
#     box_added = False
#     for gt_box in gt_boxes:
#       if box_intersect(box, gt_box) > 0:
#         if not box_added:
#           occ_area = occ_area + box_area
#           # print index
#           box.occ = 1
#           box_added=True
#           if idx%200 == 0:
#             rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,color='r',alpha=0.5)
#             ax.add_patch(rect)
#       if idx%200 == 0:
#         rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,edgecolor='r',facecolor='none')
#         ax.add_patch(rect)

#   blob_index = 0
#   for index, box in enumerate(boxes):
#     # print "Box: " + str(box.xmin) + " " + str(box.ymin) + " " + str(box.width) + " " + str(box.height)
#     row = index/feature_size
#     col = index%feature_size
#     if box.occ and not blob_intersect(box, blobs):
#       blob = grow_box(boxes, row, col, feature_size)
#       print "BLOB: xmin: " + str(blob.xmin) + " ymin: " + str(blob.ymin) + " width: " + str(blob.width) + " height: " + str(blob.height)
#       blobs.append(blob)

#   print "blobs len: " + str(len(blobs))
#   for blob in blobs:
#     blobDataDf = blobDataDf.append({'image_name':basename, 'xmin':blob.xmin, 'ymin':blob.ymin, 
#       'width':blob.width, 'height':blob.height, 
#       'area':(blob.width*blob.height)/float(total_area)}, ignore_index=True)

#   for box in boxes:
#     if blob_intersect(box, blobs) and not box.occ:
#       neigh_area = neigh_area + box_area
#       if idx%200 == 0:
#         rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,color='b',alpha=0.5)
#         ax.add_patch(rect)

#   print "Neigh Area: " + str(neigh_area/float(total_area))
#   # print "Box Area: " + str(box_area) + " Occ Area: " + str(occ_area) + " Total Area: " + str(total_area)
#   imageDataDf.loc[idx,'area_covered'] = occ_area/float(total_area)
#   imageDataDf.loc[idx,'grown_area_covered'] = neigh_area/float(total_area)
#   if idx%200 == 0:
#     fig.savefig("VerifyRegion/grown_"+str(feature_size)+"/"+basename+".jpg")
#     plt.clf()
  
#   # if(idx == 5):
#   # break

# imageDataDf.to_csv('/home/simon/DeepLearning/caffe/VerifyRegion/grown_region_'+str(feature_size)+'.csv')
# blobDataDf.to_csv('/home/simon/DeepLearning/caffe/VerifyRegion/blobs_'+str(feature_size)+'.csv')

# print "areaMean: " + str(imageDataDf['area_covered'].mean())
# print "min area: " + str(imageDataDf['area_covered'].min())
# print "max area: " + str(imageDataDf['area_covered'].max())
# print "std dev area: " + str(imageDataDf['area_covered'].std())

# print "NeighAreaMean: " + str(imageDataDf['grown_area_covered'].mean())
# print "TotalAreaCovered: " + str(imageDataDf['area_covered'].mean() + imageDataDf['grown_area_covered'].mean())