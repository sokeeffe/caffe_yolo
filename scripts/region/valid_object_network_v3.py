'''
This script processes all the images in a validation or test set for 
an object network. 
It loads up a network and tests a given dataset. 
It outputs a csv file detailing the image,cell, prediction, and ground truth 
values, from which precision recall curves can be calculated
Input Params:
 - txt file listing the valid/test dataset images (absolute path)
 - Network config prototxt
 - Network weights caffemodel
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

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
caffe_root = './'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

if len(sys.argv) != 4:
  print "Usage: python valid_object_network.py valid_images net.prototxt net.caffemodel"
  exit()

# caffe.set_mode_gpu()  
caffe.set_mode_cpu()

valid_images = sys.argv[1]
model_def = sys.argv[2]
model_weights = sys.argv[3]

output_side = 9

net = caffe.Net(model_def, model_weights, caffe.TEST)

net.blobs['data'].reshape(1,3,288,288)

out_file = '/home/simon/DeepLearning/caffe/models_bak/network_perf_07Mar2019.csv'

f = open(valid_images,'r')
images = f.readlines()
f.close()

columns = ['image_name', 'cell_index', 'conf', 'gt']
imageDataDf = pd.DataFrame(index=np.arange(len(images)*output_side*output_side),columns=columns)

class Box:
  def __init__(self, x, y, width, height, added=0):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.added = added

def overlap(x1, w1, x2, w2): #x1 ,x2 are two box xmin
    left = max(x1 - w1/2, x2 - w2/2)
    right = min(x1 + w1/2, x2 + w2/2)
    return right - left

def box_intersect(box, truth):
    w = overlap(box.x, box.width, truth.x, truth.width)
    h = overlap(box.y, box.height, truth.y, truth.height)
    if w < 0 or h < 0:
      return 0
    else:
      return 1

def darknet_resize_image(image, w, h):
  resized = np.empty((image.shape[0],h,w), dtype=np.float32)
  part = np.empty((image.shape[0], image.shape[1], w), dtype=np.float32)
  print "image Width: " + str(image.shape[2]) + " new W: " + str(w)
  w_scale = float(image.shape[2]-1)/(w-1)
  h_scale = float(image.shape[1]-1)/(h-1)
  print "W scale: " + str(w_scale) + " H scale: " + str(h_scale);
  for k in range (image.shape[0]):
    for r in range (image.shape[1]):
      for c in range(w):
        val = 0.0
        if c==w-1 or image.shape[2]==1:
          val = image[k,r,image.shape[2]-1]
        else:
          sx = c*w_scale
          ix = int(sx)
          dx = sx-ix
          val = ((1-dx)*image[k,r,ix]) + (dx*image[k,r,ix+1])
        part[k,r,c] = val
  for k in range(image.shape[0]):
    for r in range(h):
      sy = r*h_scale;
      iy = int(sy)
      dy = sy-iy
      for c in range(w):
        val = (1-dy)*part[k,iy,c]
        resized[k,r,c] = val
      if r==h-1 or image.shape[1]==1:
        continue
      for c in range(w):
        val = dy*part[k,iy+1,c]
        resized[k,r,c] += val
  return resized

def embed_image(source, dest, dx, dy):
  out = dest
  for x in range(source.shape[0]):
    for y in range(source.shape[1]):
      for k in range(source.shape[2]):
        val = source[x][y][k]
        out[x][dy+y][dx+k] = val
  return out

def letterbox_image(image, w, h):
  new_w = w
  new_h = h
  if((float(w)/image.shape[2]) < (float(h)/image.shape[1])):
    new_w = w
    new_h = (image.shape[1]*w)/image.shape[2]
  else:
    new_h = h
    new_w = (image.shape[2]*h)/image.shape[1]
  print new_w
  print new_h
  image_resized = darknet_resize_image(image, new_w, new_h)
  # image_resized = cv2.resize(image, (new_w, new_h));
  # image_resized = caffe.io.resize_image(image, [new_h, new_w])
  print image_resized.shape
  # print image_resized[0:5,0,0]
  ret = np.empty((3, 288, 288),
                           dtype=np.float32)
  ret.fill(0.5)
  out = embed_image(image_resized, ret, (w-new_w)/2, (h-new_h)/2)
  # plt.imshow(out)
  # plt.show()
  return out

def cv2_to_numpy(image):
  height = image.shape[0];
  width = image.shape[1];
  channels = image.shape[2];
  image = image[...,::-1] # BGR to RGB
  ret = np.empty((channels, height, width), dtype=np.float32)
  for h in range(height):
    for c in range(channels):
      for w in range(width):
        ret[c,h,w] = image[h,w,c]/255.0
  return ret

def det(image_index, image, basename, label_file):
  net.blobs['data'].data[...] = image
  output = net.forward()
  print "Forward Finished"

  net_output = net.blobs['object1'].data

  # Get the ground truth boxes from the label file
  gt_boxes = []
  f = open(label_file,'r')
  lines = f.readlines()
  f.close()
  for line in lines:
    box_dims = line.rstrip('\n').split(' ')
    x = float(box_dims[1])
    y = float(box_dims[2])
    width = float(box_dims[3])
    height = float(box_dims[4])
    gt_boxes.append(Box(x,y,width,height))

    output_width = net_output.shape[3]
    output_height = net_output.shape[2]
    for i in range(output_height):
      for j in range(output_width):
        box = Box((j+0.5)/float(output_width),
          (i+0.5)/float(output_height),
          (1.0/9),
          (1.0/9))
        cell_index = i*output_width + j
        df_index = image_index*output_height*output_width + cell_index
        imageDataDf.loc[df_index,'image_name'] = basename
        imageDataDf.loc[df_index,'cell_index'] = cell_index
        imageDataDf.loc[df_index,'conf'] = net_output[0,0,i,j] 
        imageDataDf.loc[df_index,'gt'] = 0
        for gt_box in gt_boxes:
          if box_intersect(box, gt_box) > 0:
            imageDataDf.loc[df_index,'gt'] = 1


for idx, image in enumerate(images):
  image_file = image.split('\n')[0]
  label_file = image_file.replace("images","labels")
  label_file = label_file.replace("png","txt")
  im = cv2.imread(image_file)
  im = cv2.resize(im,(288,288), interpolation=cv2.INTER_NEAREST)
  im = cv2_to_numpy(im)
  # im_letterbox = letterbox_image(im,288,288)
  det(idx, im, os.path.basename(image_file).split('.')[0], label_file)
  # if idx > 5:
  #   break

imageDataDf.to_csv(out_file)