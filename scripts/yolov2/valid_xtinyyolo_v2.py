# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import os
caffe_root = './'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

if len(sys.argv) != 4:
  print "Usage: python valid_xtinyyolo_v2.py valid_images net.prototxt net.caffemodel"
  exit()

caffe.set_mode_gpu()

valid_images = sys.argv[1]
model_def = sys.argv[2]
model_weights = sys.argv[3]

net = caffe.Net(model_def,
              model_weights,
              caffe.TEST)

# mu = np.array([105, 117, 123])
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          288, 288)  # image size is 227x227

# Output Files:
f_ball = open('/home/simon/DeepLearning/caffe/ball_detects.txt', 'w')
f_robot = open('/home/simon/DeepLearning/caffe/robot_detects.txt', 'w')
f_goal = open('/home/simon/DeepLearning/caffe/goal_detects.txt', 'w')
f_pen = open('/home/simon/DeepLearning/caffe/pen_detects.txt', 'w')

class Box:
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

class SortableBox:
  def __init__(self, idx, cls, probs):
    self.index = idx
    self.cls = cls
    self.probs = probs

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

def correct_region_box(box, n, w, h, net_w, net_h, relative):
  new_w = 0
  new_h = 0
  if((float(net_w)/w) < (float(net_h)/h)):
    new_w = net_w
    new_h = (h*net_w)/w
  else:
    new_h = net_h
    new_w = (w*net_h)/h
  # print "new_w: " + str(new_w) + " new_h: " + str(new_h)
  box.x = (box.x - (net_w - new_w)/2.0/net_w) / (float(new_w)/net_w)
  box.y = (box.y - (net_h - new_h)/2.0/net_h) / (float(new_h)/net_h)
  box.width = box.width*(float(net_w)/new_w)
  box.height = box.height*(float(net_h)/new_h)
  if relative:
    box.x = box.x * w
    box.y = box.y * h
    box.width = box.width * w
    box.height = box.height * h
  return box

def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box.x, box.width, truth.x, truth.width)
    h = overlap(box.y, box.height, truth.y, truth.height)
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box.width * box.height + truth.width * truth.height - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res

def do_nms(boxes, probs, total, classes, thresh):
  sortboxes=[SortableBox] * (total)
  for idx, box in enumerate(boxes):
    sortboxes[idx] = SortableBox(idx, classes, probs)
  for k in range(classes):
    for i in range(total):
      sortboxes[i].cls = k
    # print sortboxes[201].probs[201,0]
    # print sortboxes[210].probs[210,0]
    for i in range(total-1):
      for j in range(i+1,total):
        temp = SortableBox(0,0,probs)
        if sortboxes[i].probs[sortboxes[i].index,k] < sortboxes[j].probs[sortboxes[j].index,k]:
          temp = sortboxes[i]
          sortboxes[i] = sortboxes[j]
          sortboxes[j] = temp
    # print sortboxes[0].index
    for i in range(total-1):
      if probs[sortboxes[i].index, k] == 0:
        continue
      a = boxes[sortboxes[i].index]
      for j in range(i+1, total):
        b = boxes[sortboxes[j].index]
        if cal_iou(a,b) > thresh:
          probs[sortboxes[j].index, k] = 0
  return probs


def det(image,id):
  net.blobs['data'].data[...] = image
  output = net.forward()
  output_dim = net.blobs['region1'].data

  output_height = output_dim.shape[2]
  output_width = output_dim.shape[3]
  output_channels = output_dim.shape[1]
  num_boxes = 3
  coords = 4
  objectness = 1
  classes = 4
  biases = [0.31446,0.36844,  0.714939,1.156905,  1.989621,3.138732]

  boxes=[Box] * (num_boxes*output_width*output_height)
  probs=np.zeros(shape=((num_boxes*output_width*output_height), objectness+classes))

  # f = open('/home/simon/DeepLearning/caffe/region_out.csv', 'w')
  # for x in range(output_dim.shape[1]):
  #   for y in range(output_dim.shape[2]):
  #     for k in range(output_dim.shape[3]):
  #       f.write(str(output_dim[0][x][y][k]))
  #       if k < output_dim.shape[3]-1:
  #         f.write(',')
  #     f.write('\n')
  # f.close()

  # Equivalent to get_region_boxes()
  for i in range(output_width*output_height):
    row = i/output_width
    col = i%output_height
    for n in range(num_boxes):
      index = n*output_width*output_height + i
      obj_index = n*(coords+objectness+classes) + coords
      box_index = n*(coords+objectness+classes)
      class_index = n*(coords+objectness+classes) + coords + objectness
      scale = output_dim[0,obj_index, row, col]
      boxes[index] = Box((col + output_dim[0,box_index,row,col])/output_width,
        (row + output_dim[0,box_index+1,row,col])/output_width,
        np.exp(output_dim[0,box_index+2,row,col]) * biases[2*n] / output_width,
        np.exp(output_dim[0,box_index+3,row,col]) * biases[2*n + 1] / output_height)
      ball_prob = scale*output_dim[0,class_index,row,col]
      if ball_prob < 0.005:
        ball_prob = 0
      probs[index,0] = ball_prob
      robot_prob = scale*output_dim[0,class_index+1,row,col]
      if robot_prob < 0.005:
        robot_prob = 0
      probs[index,1] = robot_prob
      goal_prob = scale*output_dim[0,class_index+2,row,col]
      if goal_prob < 0.005:
        goal_prob = 0
      probs[index,2] = goal_prob
      pen_prob = scale*output_dim[0,class_index+3,row,col]
      if pen_prob < 0.005:
        pen_prob = 0
      probs[index,3] = pen_prob
      probs[index,4] = max(probs[index,0:4])
      # if probs[index,4] > 0.8:
      #   print "Index: " + str(index)

  probs = do_nms(boxes, probs, num_boxes*output_width*output_height, classes, 0.45) 

  for indx, box in enumerate(boxes):
    box = correct_region_box(box, (output_width*output_height*num_boxes), 640, 480, 288, 288, 1)
    xmin = box.x - box.width/2 + 1
    ymin = box.y - box.height/2 + 1
    xmax = box.x + box.width/2 + 1
    ymax = box.y + box.height/2 + 1

    if(xmin < 1):
      xmin = 1
    if(ymin < 1):
      ymin = 1
    if(xmax > 640):
      xmax = 640
    if(ymax > 480):
      ymax = 480
    if probs[indx,4]>0.005:
      if probs[indx,4] == probs[indx,0]:
        f_ball.write(id + " " + str(probs[indx,0]) + " " + str(xmin)
                + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
      if probs[indx,4] == probs[indx,1]:
        f_robot.write(id + " " + str(probs[indx,1]) + " " + str(xmin)
                + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
      if probs[indx,4] == probs[indx,2]:
        f_goal.write(id + " " + str(probs[indx,2]) + " " + str(xmin)
                + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
      if probs[indx,4] == probs[indx,3]:
        f_pen.write(id + " " + str(probs[indx,3]) + " " + str(xmin)
                + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")  

f = open(valid_images,'r')
images = f.readlines()
for image in images:
  image_file = image.split('\n')[0]
  im = cv2.imread(image_file)
  im = cv2_to_numpy(im)
  im_letterbox = letterbox_image(im, 288, 288)
  det(im_letterbox, os.path.basename(image_file).split('.')[0])
  # break
f_ball.close()
f_robot.close()
f_goal.close()
f_pen.close()