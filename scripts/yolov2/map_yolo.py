#! /usr/bin/env python
''' Reads in the precision and recall points for each of the four classes
    in the SPL dataset. It calculates the average precision for each class and 
    then the mean average precision over all classes.
    
    @param ball precision recall data
    @param robot precision recall data
    @param goalpost precision recall data
    @param penspot precision recall data
    
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

if len(sys.argv) != 6:
	print "Usage: ./map_yolo.py ball_pr_data robot_pr_data goalpost_pr_data penspot_pr_data outfile"
	exit()

##### CHANGE THESE PATHS FOR DEPENDING ON DATASET
ball_pr_data = pd.read_csv(sys.argv[1])
robot_pr_data = pd.read_csv(sys.argv[2])
goalpost_pr_data = pd.read_csv(sys.argv[3])
penspot_pr_data = pd.read_csv(sys.argv[4])

# Sort our pr dataframes with ascending recall for plotting.
ball_pr_data = ball_pr_data.sort(['Recall'],ascending=[1])
robot_pr_data = robot_pr_data.sort(['Recall'],ascending=[1])
goalpost_pr_data = goalpost_pr_data.sort(['Recall'],ascending=[1])
penspot_pr_data = penspot_pr_data.sort(['Recall'],ascending=[1])

recallRange = np.arange(0.0, 1.01, 0.1)
ball_precisions = []
robot_precisions = []
goalpost_precisions = []
penspot_precisions = []

for recallVal in recallRange:
	ball_precisions.append(ball_pr_data.loc[ball_pr_data['Recall'] >= recallVal]['Precision'].max())
	robot_precisions.append(robot_pr_data.loc[robot_pr_data['Recall'] >= recallVal]['Precision'].max())
	goalpost_precisions.append(goalpost_pr_data.loc[goalpost_pr_data['Recall'] >= recallVal]['Precision'].max())
	penspot_precisions.append(penspot_pr_data.loc[penspot_pr_data['Recall'] >= recallVal]['Precision'].max())

ball_precisions = [0 if math.isnan(x) else x for x in ball_precisions]	
robot_precisions = [0 if math.isnan(x) else x for x in robot_precisions]	
goalpost_precisions = [0 if math.isnan(x) else x for x in goalpost_precisions]
penspot_precisions = [0 if math.isnan(x) else x for x in penspot_precisions]	

ball_ap = np.mean(ball_precisions)
robot_ap = np.mean(robot_precisions)
goalpost_ap = np.mean(goalpost_precisions)
penspot_ap = np.mean(penspot_precisions)

mAP = np.mean([ball_ap, robot_ap, goalpost_ap, penspot_ap])

#~ print "Ball Average Precision: " + str(ball_ap)
#~ print "Robot Average Precision: " + str(robot_ap)
#~ print "Goal Post Average Precision: " + str(goalpost_ap)
#~ print "Pen Spot Average Precision: " + str(penspot_ap)
#~ print "Mean Average Precision: " + str(mAP)

f = open(sys.argv[5], 'w')
f.write(str(ball_ap)+'\n'+str(robot_ap)+'\n'+str(goalpost_ap)+'\n'+str(penspot_ap)+'\n'+str(mAP))
f.close()

plt.show()

