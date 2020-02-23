'''
This script analyses the SPLObjDetect dataset and predicts the amount of
image covered with a possible grid based region proposal
It considers the image coverage for different feature map spatial dimensions
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

if len(sys.argv) < 2:
  print "Usage python eval_region_cover.py dataset_list.txt"
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

# ret = np.empty((288, 288, 3), dtype=np.float32)
# ret.fill(0.5)

# fig, ax = plt.subplots(1)
# ax.imshow(ret)
# for box in boxes:
#   rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,edgecolor='r',facecolor='none')
#   ax.add_patch(rect)

# plt.show()

occ_areas=[]

image_file = sys.argv[1]
f = open(image_file,'r')
image_files = f.readlines()
f.close()

columns = ['image_name', 'num_objects','area_covered']
imageDataDf = pd.DataFrame(index=np.arange(len(image_files)),columns=columns)

for idx, line in enumerate(image_files):
  image_file = line.split('\n')[0]
  basename = os.path.basename(image_file).split('.')[0]
  label_file = image_file.replace("images","labels")
  label_file = label_file.replace("png","txt")

  imageDataDf.loc[idx,'image_name'] = basename

  #if idx%200 == 0:
  im = np.array(Image.open(image_file).resize((288,288), Image.ANTIALIAS))
  fig, ax = plt.subplots(1)
  ax.imshow(im)

  gt_boxes = []
  f = open(label_file,'r')
  lines = f.readlines()
  f.close()
  for line in lines:
    box_dims = line.rstrip('\n').split(' ')
    xmin = int((float(box_dims[1])-float(box_dims[3])/2.0)*image_dim)
    ymin = int((float(box_dims[2])-float(box_dims[4])/2.0)*image_dim)
    width = int(float(box_dims[3])*image_dim)
    height = int(float(box_dims[4])*image_dim)
    gt_boxes.append(Box(xmin,ymin,width,height))

  imageDataDf.loc[idx,'num_objects'] = len(lines)

  box_area = pool_factor*pool_factor
  occ_area = 0
  total_area = image_dim*image_dim

  for box in boxes:
    box_added = False
    for gt_box in gt_boxes:
      if box_intersect(box, gt_box) > 0:
        if not box_added:
          occ_area = occ_area + box_area
          box_added=True
        #if idx%200 == 0:
        rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,color='r',alpha=0.5)
        ax.add_patch(rect)
      #if idx%200 == 0:
      rect = patches.Rectangle((box.xmin, box.ymin),box.width,box.height,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)

  # print "Box Area: " + str(box_area) + " Occ Area: " + str(occ_area) + " Total Area: " + str(total_area)
  imageDataDf.loc[idx,'area_covered'] = occ_area/float(total_area)
  #if idx%200 == 0:
  fig.savefig("VerifyKITTI/"+str(feature_size)+"/"+basename+".jpg")
  plt.clf()

imageDataDf.to_csv('/home/simon/DeepLearning/caffe/VerifyKITTI/region_'+str(feature_size)+'.csv')

print "areaMean: " + str(imageDataDf['area_covered'].mean())
print "min area: " + str(imageDataDf['area_covered'].min())
print "max area: " + str(imageDataDf['area_covered'].max())
print "std dev area: " + str(imageDataDf['area_covered'].std())
