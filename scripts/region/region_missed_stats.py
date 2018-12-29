'''
This script reads in analyses of the SPLObjDetect dataset and outputs 
some statistics on the image covered objects using a grid based region proposal
It considers the image coverage for different feature map spatial dimensions
and connected cells are considered in regtangular blobs
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

if len(sys.argv) < 3:
  print "Usage python region_network_stats.py dataset_region.csv out_file.csv"

regionDf = pd.read_csv(sys.argv[1])
outputDf_file = sys.argv[2]

regionDf['missed_objects'] = regionDf['gt_objects'] - regionDf['detect_objects'] - regionDf['part_objects'] - regionDf['bit_objects']
regionDf.loc[regionDf['missed_objects']<0,'missed_objects'] = 0

regionDf['missed_ball'] = regionDf['gt_ball'] - regionDf['detect_ball'] - regionDf['part_ball'] - regionDf['bit_ball']
regionDf.loc[regionDf['missed_ball']<0,'missed_ball'] = 0

regionDf['missed_robot'] = regionDf['gt_robot'] - regionDf['detect_robot'] - regionDf['part_robot'] - regionDf['bit_robot']
regionDf.loc[regionDf['missed_robot']<0,'missed_robot'] = 0

regionDf['missed_goalpost'] = regionDf['gt_goalpost'] - regionDf['detect_goalpost'] - regionDf['part_goalpost'] - regionDf['bit_goalpost']
regionDf.loc[regionDf['missed_goalpost']<0,'missed_goalpost'] = 0

regionDf['missed_penspot'] = regionDf['gt_penspot'] - regionDf['detect_penspot'] - regionDf['part_penspot'] - regionDf['bit_penspot']
regionDf.loc[regionDf['missed_penspot']<0,'missed_penspot'] = 0

print "Area Covered Stats:"
print "\tMean: " + str(regionDf['area_covered'].mean())
print "\tMax: " + str(regionDf['area_covered'].max())
print "\tMin: " + str(regionDf['area_covered'].min())
print "\tStd Dev: " + str(regionDf['area_covered'].std())

print
print "Missed Objects:"
print "\tTotal: " + str(regionDf['missed_objects'].sum())
print "\tPrct: " + str(float(regionDf['missed_objects'].sum()) / regionDf['gt_objects'].sum())

print
print "Missed Ball:"
print "\tTotal: " + str(regionDf['missed_ball'].sum())
print "\tPrct: " + str(float(regionDf['missed_ball'].sum()) / regionDf['gt_ball'].sum())

print
print "Missed Objects:"
print "\tTotal: " + str(regionDf['missed_robot'].sum())
print "\tPrct: " + str(float(regionDf['missed_robot'].sum()) / regionDf['gt_robot'].sum())

print
print "Missed Objects:"
print "\tTotal: " + str(regionDf['missed_goalpost'].sum())
print "\tPrct: " + str(float(regionDf['missed_goalpost'].sum()) / regionDf['gt_goalpost'].sum())

print
print "Missed Objects:"
print "\tTotal: " + str(regionDf['missed_penspot'].sum())
print "\tPrct: " + str(float(regionDf['missed_penspot'].sum()) / regionDf['gt_penspot'].sum())

columns = ['class','class_index','missed']
missedObjectsDf = pd.DataFrame(index=range(4),columns=columns)
missedObjectsDf.loc[0,'class'] = 'ball'
missedObjectsDf.loc[1,'class'] = 'robot'
missedObjectsDf.loc[2,'class'] = 'goalpost'
missedObjectsDf.loc[3,'class'] = 'penspot'
missedObjectsDf.loc[0,'class_index'] = 0
missedObjectsDf.loc[1,'class_index'] = 1
missedObjectsDf.loc[2,'class_index'] = 2
missedObjectsDf.loc[3,'class_index'] = 3
missedObjectsDf.loc[0,'missed'] = regionDf['missed_ball'].sum()
missedObjectsDf.loc[1,'missed'] = regionDf['missed_robot'].sum()
missedObjectsDf.loc[2,'missed'] = regionDf['missed_goalpost'].sum()
missedObjectsDf.loc[3,'missed'] = regionDf['missed_penspot'].sum()

total_recall = (regionDf['gt_objects'].sum()-regionDf['missed_objects'].sum())/float(regionDf['gt_objects'].sum())
object_part_recall = (regionDf['detect_objects'].sum()+regionDf['part_objects'].sum())/float(regionDf['gt_objects'].sum())
object_recall = (regionDf['detect_objects'].sum())/float(regionDf['gt_objects'].sum())

print "Total Recall: " + str(total_recall)
print "Object Recall: " + str(object_recall)
print "Object Part Recall: " + str(object_part_recall)

# missedObjectsDf.to_csv(outputDf_file,index=False)