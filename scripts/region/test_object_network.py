# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = './'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

if len(sys.argv) != 4:
  print "Usage: python test_object_network.py net.prototxt net.caffemodel testImage.png"
  exit()

#************************************************************************************************8

# caffe.set_mode_gpu()
caffe.set_mode_cpu()

model_def = sys.argv[1]
model_weights = sys.argv[2]

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

class Box:
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

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

def det(image, image_id, pic):
  net.blobs['data'].data[...] = image
  output = net.forward()
  print "Forward Finished"

  net_output = net.blobs['object1'].data
  print net_output.shape

  output_width = net_output.shape[3]
  output_height = net_output.shape[2]
  im = cv2.imread(pic)
  color = (255, 242, 35)
  for i in range(output_height):
    for j in range(output_width):
      # print net_output[0,0,i,j],
      if net_output[0,0,i,j] > 0.5:
        box = Box(int(((j+0.5)/float(output_width))*640),
          int(((i+0.5)/float(output_height))*480),
          int(640/9),
          int(480/9))
        print "i: " + str(i) + " j: " + str(j)
        print "Box x: "+str(box.x)+" y: "+str(box.y)+" w: "+str(box.width)+" h: "+str(box.height)
        overlay = im.copy()
        cv2.rectangle(overlay,(int(box.x - (box.width/2)), 
          int(box.y - (box.height/2))),
          (int(box.x + (box.width/2)),
          int(box.y + (box.height/2))),(0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, im, 1-0.4, 0, im)
    print
  
  cv2.imshow('src', im)
  cv2.waitKey()

pic = sys.argv[3]

image = cv2.imread(pic)
image = cv2_to_numpy(image)

image_letterbox = letterbox_image(image, 288, 288)

det(image_letterbox,'10001',pic)
print "Done"