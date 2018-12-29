'''
This script processes all the images in a validation or test set for 
an object network. 
It loads up a network and tests a given dataset. 
It outputs a the detected regions as new images along with the ground truth objects
for further processing. Every image patch is saved under the original image name
with an index number appended to it. The format of the labels is the same as 
the darknet framework
Input Params:
 - txt file listing the valid/test dataset images (absolute path)
 - Network config prototxt
 - Network weights caffemodel
 - Output Directory
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

if len(sys.argv) != 5:
  print "Usage: python extract_detections.py valid_images net.prototxt net.caffemodel out_directory"
  exit()

caffe.set_mode_gpu()  

valid_images = sys.argv[1]
model_def = sys.argv[2]
model_weights = sys.argv[3]
out_dir = sys.argv[4]

output_side = 9

columns = ['image_name', 'area_covered', 'gt_objects', 
          'detect_objects','part_objects','bit_objects',
          'gt_ball', 'detect_ball', 'part_ball','bit_ball',
          'gt_robot','detect_robot','part_robot','bit_robot',
          'gt_goalpost','detect_goalpost','part_goalpost','bit_goalpost',
          'gt_penspot','detect_penspot','part_penspot','bit_penspot']
imageDataDf = pd.DataFrame(columns=columns)

net = caffe.Net(model_def, model_weights, caffe.TEST)

net.blobs['data'].reshape(1,3,288,288)

f = open(valid_images,'r')
images = f.readlines()
f.close()

class Box:
  def __init__(self, x, y, width, height, clss = 0, added=0, occ=0, neigh=0,blob=-1):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.clss = clss
    self.xmin = x - width/2.
    self.ymin = y - height/2.
    self.added = added
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
    left = max(x1 - w1/2, x2 - w2/2)
    right = min(x1 + w1/2, x2 + w2/2)
    return right - left

def xmin_overlap(x1, w1, x2, w2): #x1 ,x2 are two box xmin
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    return right - left

def box_intersect(box, truth):
    w = overlap(box.x, box.width, truth.x, truth.width)
    h = overlap(box.y, box.height, truth.y, truth.height)
    if w < 0 or h < 0:
      return 0
    area = w*h
    return area

def blob_intersect(box, blobs):
  if not blobs:
    return False
  for blob in blobs:
    w = xmin_overlap(box.xmin, box.width, blob.xmin, blob.width)
    h = xmin_overlap(box.ymin, box.height, blob.ymin, blob.height)
    if w > 0 and h > 0:
      return 1
  return 0

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
  for r in range(min_row, max_row+1):
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
  # print "Xmin: " + str(xmin) + " YMin: " + str(ymin) + " XMax: " + str(xmax) +" YMax: " + str(ymax)
  return Blob(xmin, ymin, xmax-xmin, ymax-ymin)

def det(image_index, image, basename, label_file, image_file):
  net.blobs['data'].data[...] = image
  output = net.forward()
  print "Forward Finished"

  imageDataDf.loc[image_index,'image_name'] = basename
  imageDataDf.loc[image_index,'area_covered'] = 0.0
  imageDataDf.loc[image_index,'detect_objects'] = 0
  imageDataDf.loc[image_index,'part_objects'] = 0
  imageDataDf.loc[image_index,'bit_objects'] = 0
  imageDataDf.loc[image_index,'gt_ball'] = 0
  imageDataDf.loc[image_index,'detect_ball'] = 0
  imageDataDf.loc[image_index,'part_ball'] = 0
  imageDataDf.loc[image_index,'bit_ball'] = 0
  imageDataDf.loc[image_index,'gt_robot'] = 0
  imageDataDf.loc[image_index,'detect_robot'] = 0
  imageDataDf.loc[image_index,'part_robot'] = 0
  imageDataDf.loc[image_index,'bit_robot'] = 0
  imageDataDf.loc[image_index,'gt_goalpost'] = 0
  imageDataDf.loc[image_index,'detect_goalpost'] = 0
  imageDataDf.loc[image_index,'part_goalpost'] = 0
  imageDataDf.loc[image_index,'bit_goalpost'] = 0
  imageDataDf.loc[image_index,'gt_penspot'] = 0
  imageDataDf.loc[image_index,'detect_penspot'] = 0
  imageDataDf.loc[image_index,'part_penspot'] = 0
  imageDataDf.loc[image_index,'bit_penspot'] = 0

  net_output = net.blobs['object1'].data
  output_width = net_output.shape[3]
  output_height = net_output.shape[2]
  pool_factor = image.shape[2]/output_width

  thresh = 0.5
  blobs = []

  boxes = [Box] * (output_width * output_height)
  for i in range(output_height):
    for j in range(output_width):
      y = (i+0.5)/output_height
      x = (j+0.5)/output_width
      index = i*output_height + j
      boxes[index] = Box(x, y, 1./output_width, 1./output_height)
      if net_output[0,0,i,j] > thresh:
        # print "Obj x: " + str(j) + " y: " + str(i) + " " + str(x) + " " + str(y) + " " + str(1./output_width)
        boxes[index].occ = 1

  # Get the ground truth boxes from the label file
  gt_boxes = []
  f = open(label_file,'r')
  lines = f.readlines()
  imageDataDf.loc[image_index,'gt_objects'] = len(lines)
  f.close()
  for line in lines:
    box_dims = line.rstrip('\n').split(' ')
    x = float(box_dims[1])
    y = float(box_dims[2])
    width = float(box_dims[3])
    height = float(box_dims[4])
    if int(box_dims[0]) == 0:
      imageDataDf.loc[image_index,'gt_ball'] = imageDataDf.loc[image_index,'gt_ball'] + 1
    elif int(box_dims[0])==1:
      imageDataDf.loc[image_index,'gt_robot'] = imageDataDf.loc[image_index,'gt_robot'] + 1
    elif int(box_dims[0])==2:
      imageDataDf.loc[image_index,'gt_goalpost'] = imageDataDf.loc[image_index,'gt_goalpost'] + 1  
    elif int(box_dims[0])==3:
      imageDataDf.loc[image_index,'gt_penspot'] = imageDataDf.loc[image_index,'gt_penspot'] + 1
    gt_boxes.append(Box(x,y,width,height, int(box_dims[0])))

  for index, box in enumerate(boxes):
    row = index/output_height
    col = index%output_height
    if box.occ and not blob_intersect(box, blobs):
      blob = grow_box(boxes, row, col, output_width)
      # print "BLOB: xmin: " + str(blob.xmin) + " ymin: " + str(blob.ymin) + " width: " + str(blob.width) + " height: " + str(blob.height)
      blobs.append(blob)

  im = Image.open(image_file)
  for index, blob in enumerate(blobs):
    # print "Blob Norm left: " + str(blob.xmin) + " top: " + str(blob.ymin) + " right: " + str(blob.xmin+blob.width) + " bottom: " + str(blob.ymin+blob.height)
    height = np.array(im).shape[0]
    width = np.array(im).shape[1]
    # print width
    # print height
    imageDataDf.loc[image_index,'area_covered'] = imageDataDf.loc[image_index, 'area_covered'] + (blob.width*blob.height)
    im_left = blob.xmin*width
    im_top = blob.ymin*height
    im_right = (blob.xmin+blob.width)*width
    im_bottom = (blob.ymin+blob.height)*height
    im_patch = im.crop((int(im_left), int(im_top),int(im_right), int(im_bottom)))
    im_outfile = out_dir + "/train/images/" + basename + "_" + str(index) + ".png"
    # im_patch.save(im_outfile, "PNG")
    # print os.path.abspath(im_outfile)
    # with open(out_dir+"/train.txt", 'a') as f:
    #   f.write(os.path.abspath(im_outfile)+'\n')
    label_outfile = out_dir + "/train/labels/" + basename + "_" + str(index) + ".txt"
    for gt_box in gt_boxes:
      area_overlap = box_intersect(gt_box, Box(blob.xmin+blob.width/2.,blob.ymin+blob.height/2.,blob.width, blob.height))
      if area_overlap:
        # print "Blob Intersect: " + str(area_overlap) + " " + str(gt_box.width*gt_box.height)
        area_covered = area_overlap/(gt_box.width*gt_box.height)
        normX = (gt_box.x - blob.xmin) / blob.width
        normY = (gt_box.y - blob.ymin) / blob.height
        normWidth = gt_box.width / blob.width
        normHeight = gt_box.height / blob.height
        if area_covered > 0.95:
          imageDataDf.loc[image_index,'detect_objects'] = imageDataDf.loc[image_index,'detect_objects'] + 1
          if gt_box.clss == 0:
            imageDataDf.loc[image_index,'detect_ball'] = imageDataDf.loc[image_index,'detect_ball'] + 1
          elif gt_box.clss == 1:
            imageDataDf.loc[image_index,'detect_robot'] = imageDataDf.loc[image_index,'detect_robot'] + 1
          elif gt_box.clss == 2:
            imageDataDf.loc[image_index,'detect_goalpost'] = imageDataDf.loc[image_index,'detect_goalpost'] + 1
          elif gt_box.clss == 3:
            imageDataDf.loc[image_index,'detect_penspot'] = imageDataDf.loc[image_index,'detect_penspot'] + 1
        elif area_covered > 0.2:
          imageDataDf.loc[image_index,'part_objects'] = imageDataDf.loc[image_index,'part_objects'] + 1
          if gt_box.clss == 0:
            imageDataDf.loc[image_index,'part_ball'] = imageDataDf.loc[image_index,'part_ball'] + 1
          elif gt_box.clss == 1:
            imageDataDf.loc[image_index,'part_robot'] = imageDataDf.loc[image_index,'part_robot'] + 1
          elif gt_box.clss == 2:
            imageDataDf.loc[image_index,'part_goalpost'] = imageDataDf.loc[image_index,'part_goalpost'] + 1
          elif gt_box.clss == 3:
            imageDataDf.loc[image_index,'part_penspot'] = imageDataDf.loc[image_index,'part_penspot'] + 1
        else:
          imageDataDf.loc[image_index,'bit_objects'] = imageDataDf.loc[image_index,'bit_objects'] + 1
          if gt_box.clss == 0:
            imageDataDf.loc[image_index,'bit_ball'] = imageDataDf.loc[image_index,'bit_ball'] + 1
          elif gt_box.clss == 1:
            imageDataDf.loc[image_index,'bit_robot'] = imageDataDf.loc[image_index,'bit_robot'] + 1
          elif gt_box.clss == 2:
            imageDataDf.loc[image_index,'bit_goalpost'] = imageDataDf.loc[image_index,'bit_goalpost'] + 1
          elif gt_box.clss == 3:
            imageDataDf.loc[image_index,'bit_penspot'] = imageDataDf.loc[image_index,'bit_penspot'] + 1
        # with open(label_outfile, 'a') as f:
        #   f.write(str(gt_box.clss)+" "+str(normX)+" "+str(normY)+" "+str(normWidth)+" "+str(normHeight)+'\n')

  # for i in range(output_height):
  #   for j in range(output_width):
  #     box = Box((j+0.5)/float(output_width),
  #       (i+0.5)/float(output_height),
  #       (1.0/output_widht),
  #       (1.0/output_height))
  #     cell_index = i*output_width + j
  #     df_index = image_index*output_height*output_width + cell_index
  #     imageDataDf.loc[df_index,'image_name'] = basename
  #     imageDataDf.loc[df_index,'cell_index'] = cell_index
  #     imageDataDf.loc[df_index,'conf'] = net_output[0,0,i,j] 
  #     imageDataDf.loc[df_index,'gt'] = 0
  #     for gt_box in gt_boxes:
  #       box_inter = box_intersect(box, gt_box)
  #       if box_inter > 0:
  #         # print "Box width: " + str(box.width) + " height: " + str(box.height)
  #         # print "Box width: " + str(gt_box.width) + " height: " + str(gt_box.height)
  #         # print "Box Intersect: " + str(box_inter)
  #         imageDataDf.loc[df_index,'gt'] = float(box_inter)/(box.width*box.height)


for idx, image in enumerate(images):
  image_file = image.split('\n')[0]
  label_file = image_file.replace("images","labels")
  label_file = label_file.replace("png","txt")
  print image_file
  im = cv2.imread(image_file)
  im = cv2.resize(im,(288,288), interpolation=cv2.INTER_NEAREST)
  im = cv2_to_numpy(im)
  # im_letterbox = letterbox_image(im,288,288)
  det(idx, im, os.path.basename(image_file).split('.')[0], label_file, image_file)
  # if idx > 3:
  #   break

out_file = "results/region_detector_val_out.csv"
imageDataDf.to_csv(out_file)