# This script performs inference on one image on an input of 64x64
#
# @author: Simon O'Keeffe
# @date: 01Feb2020



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
	print "Usage: python test_yolo_region_rp9.py net.prototxt net.caffemodel testImage.png"
	exit()

#************************************************************************************************8

# caffe.set_mode_gpu()
caffe.set_mode_cpu()

model_def = sys.argv[1]
model_weights = sys.argv[2]

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([105, 117, 123])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          64, 64)  # image size is 64x64

#************************************************************************************************

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
	ret = np.empty((3, 64, 64),
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

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
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


def det(image, image_id, pic):

	# Transform the letterbox image into Caffe format. Do not perform if working off CSV file
	# transpose = transformer.transpose.get('data')
	# channel_swap = transformer.channel_swap.get('data')
	# temp = image.transpose(transpose)
	# temp = temp[channel_swap, :, :]

	# Use temp for image processing
	# net.blobs['data'].data[...] = temp
	net.blobs['data'].data[...] = image

	print "Forward Started"
	output = net.forward()
	print "Forward Finished"
	first_conv = net.blobs['scale1'].data
	print first_conv.shape 

	input_image = net.blobs['data'].data
	print input_image.shape
	output_dim = net.blobs['region1'].data
	print output_dim.shape

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
			probs[index,0] = scale*output_dim[0,class_index,row,col]
			probs[index,1] = scale*output_dim[0,class_index+1,row,col]
			probs[index,2] = scale*output_dim[0,class_index+2,row,col]
			probs[index,3] = scale*output_dim[0,class_index+3,row,col]
			probs[index,4] = max(probs[index,0:4])
			if scale > 0.728	:
				print obj_index
				print box_index
				print scale
				print index
				print "Box X: " + str(boxes[index].x) + " Y: " + str(boxes[index].y) + " W: " + str(boxes[index].width) + " H: " + str(boxes[index].height)
				print "Probs Ball: " + str(probs[index,0]) + " Robot: " + str(probs[index,1]) + " Goal Post: " + str(probs[index,2]) + " Pen Spot: " + str(probs[index,3]) + " Max: " + str(probs[index,4])

	for box in boxes:
		box = correct_region_box(box, (output_width*output_height*num_boxes), 640, 480, 288, 288, 1)
	print "Box X: " + str(boxes[202].x) + " Y: " + str(boxes[202].y) + " W: " + str(boxes[202].width) + " H: " + str(boxes[202].height)

	im = cv2.imread(pic)
	color = (255, 242, 35)
	for index, box in enumerate(boxes):
		if probs[index, 4] > 0.2:
			cv2.rectangle(im,(int(box.x - (box.width/2)), int(box.y - (box.height/2))),(int(box.x + (box.width/2)),int(box.y + (box.height/2))),color,3)
	
	cv2.imshow('src', im)
	cv2.waitKey()
	print 'det'


pic = sys.argv[3]


image2 = cv2.imread(pic)
image2 = cv2_to_numpy(image2)
print image2.shape

image_letterbox2 = letterbox_image(image2, 64, 64)
f = open('/home/simon/DeepLearning/caffe/letterbox_image.csv', 'w')
for x in range(image_letterbox2.shape[0]):
	for y in range(image_letterbox2.shape[1]):
		for k in range(image_letterbox2.shape[2]):
			f.write(str(image_letterbox2[x][y][k]))
			if k < image_letterbox2.shape[2] - 1:
				f.write(',')
		f.write('\n')
f.close()

det(image_letterbox2,'10001',pic)
print 'over'

