#! /usr/bin/env python
''' Reads in the detection for results for a yolo network along with the
    ground truth and calculates the precision and recall for various levels
    of confidence. It then plots this precision-recall curve and outputs 
    these precision recall values to a specified csv file.
    
    @param detection results and confidence for a given class
    @param ground truth results for a given class
    @param outfile to write precision recall values too
    
    @author: Simon
    @date: 7 Dec 2017'''
import sys
import getopt
import os
import errno
import math
import glob
import copy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def overlap(b1x1, b1x2, b2x1, b2x2):
	left = b1x1
	if b1x1 < b2x1: 
		left = b2x1
	right = b1x2
	if b1x2 > b2x2:
		right = b2x2
	return right - left
	
def box_intersection(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
	w = overlap(b1x1, b1x2, b2x1, b2x2)
	h = overlap(b1y1, b1y2, b2y1, b2y2)
	if (w < 0 or h < 0):
		return 0
	area = w*h
	return area
	
def box_union(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
	i = box_intersection(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
	w1 = b1x2 - b1x1
	h1 = b1y2 - b1y1
	w2 = b2x2 - b2x1
	h2 = b2y2 - b2y1
	u = w1*h1 + w2*h2 - i
	return u
	
def box_iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
	iou = box_intersection(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)/box_union(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
	return iou

if len(sys.argv) != 6:
	print "Usage: ./meanAveragePrecision.py detections.txt groundTruth.txt regionMisses.csv class outfile"
	exit()

###### CHANGE INPUT DIRECTORY PATH FOR RESULT HERE
results_df = pd.read_csv(sys.argv[1],' ',header=None)
results_df.columns = ["Image","Prob","x1","y1","x2","y2"]
# print results_df

###### CHANGE INPUT DIRECTORY PATH FOR GROUND TRUTH HERE
gt_df = pd.read_csv(sys.argv[2])
# print gt_df
gt_df = gt_df.loc[gt_df['Class']==int(sys.argv[4])]
gt_df['x1'] = (gt_df['x'] - gt_df['w']/2.)
gt_df.loc[gt_df['x1']<0, 'x1'] = 0
gt_df['y1'] = (gt_df['y'] - gt_df['h']/2.)
gt_df.loc[gt_df['y1']<0, 'y1'] = 0
gt_df['x2'] = (gt_df['x'] + gt_df['w']/2.)
gt_df.loc[gt_df['x2']>=1, 'x2'] = 1
gt_df['y2'] = (gt_df['y'] + gt_df['h']/2.)
gt_df.loc[gt_df['y2']>=1, 'y2'] = 1
gt_df['detected'] = 0
gt_df['Prob'] = 0

gt_df.reset_index()
# print gt_df

missed_df = pd.read_csv(sys.argv[3])

##### SPECIFY DESIRED IOU VALUE HERE
IOU_thresh = 0.5

true_positives = 0
false_positives = 0
false_negatives = 0

##### CHANGE THRESHOLD RANGE FOR MORE GRANULARITY IN PRECISION RECALL CURVE
thresholdRange = np.arange(0.02, 1.0, 0.02)
thresholdRange = np.insert(thresholdRange, 0, [0.001, 0.01])
pr_df = []

for index, row in results_df.iterrows():
	image = row['Image']
	correctDetection = False
	# image = image + '.png'
	# print gt_df.loc[gt_df['Image']==image]
	for index2, row2 in gt_df.loc[gt_df['Image']==image].iterrows():
		iou = box_iou(row['x1'],row['y1'],row['x2'],row['y2'],row2['x1'],row2['y1'],row2['x2'],row2['y2'])
		# print iou
		if iou > IOU_thresh:
			gt_df.loc[index2,'detected'] = 1
			if row['Prob'] > row2['Prob']:
				gt_df.loc[index2,'Prob'] = row['Prob']
			correctDetection = True
	# break

# print gt_df

for idx, thresh in enumerate(thresholdRange):
	true_positives = gt_df.loc[gt_df['detected']==1].loc[gt_df['Prob'] >= thresh].shape[0]
	total_positives = results_df.loc[results_df['Prob'] >= thresh].shape[0]
	# Add Completely Misses Objects here to total objects total
	total_objects = gt_df.shape[0]+missed_df.loc[int(sys.argv[4]),'missed']
	# print missed_df.loc[int(sys.argv[4]),'missed']
	if (total_positives) > 0:
		precision = float(true_positives)/float(total_positives)
	else:
		precision = 1.0
	if (total_objects) > 0:
		recall = float(true_positives)/float(total_objects)
	else:
		recall = 0.0
	d = {'Precision': [precision], 'Recall': [recall]}
	if idx == 0:
		pr_df = pd.DataFrame(d)
	else:
		pr_df2 = pd.DataFrame(d)
		pr_df = pr_df.append(pr_df2, ignore_index=True,verify_integrity=True)

################################################################################
#for idx, thresh in enumerate(thresholdRange):
#	true_positives = 0
#	false_positives = 0
#	false_negatives = 0
#	for index, row in results_df.iterrows():
#		if row['Prob'] < thresh:
#			continue
#		image = row['Image']
#		correctDetection = False
#		for index2, row2 in gt_df.loc[gt_df['Image']==image].iterrows():
#			iou = box_iou(row['x1'],row['y1'],row['x2'],row['y2'],row2['x1'],row2['y1'],row2['x2'],row2['y2'])
#			if iou > IOU_thresh and row2['detected'] == 0:
#				true_positives=true_positives+1
#				gt_df.loc[index2,'detected'] = 1
#				correctDetection = True
#		if correctDetection == False:
#			false_positives = false_positives+1
#	false_negatives = gt_df.loc[gt_df['detected'] == 0].count()['detected']
#	if (true_positives+false_positives) > 0:
#		precision = true_positives/float(true_positives+false_positives)
#	else:
#		precision = 1.0
#	if (true_positives+false_negatives) > 0:
#		recall = true_positives/float(true_positives+false_negatives)
#	else:
#		recall = 0.0
#	d = {'Precision': [precision], 'Recall': [recall]}
#	if idx == 0:
#		pr_df = pd.DataFrame(d)
#	else:
#		pr_df2 = pd.DataFrame(d)
#		pr_df = pr_df.append(pr_df2, ignore_index=True,verify_integrity=True)
#	gt_df['detected'] = 0


#print pr_df
##### CHANGE OUTPUT FILE FOR PRECISION RECALL POINTS
################################################################################

pr_df.to_csv(sys.argv[5],index=False)
