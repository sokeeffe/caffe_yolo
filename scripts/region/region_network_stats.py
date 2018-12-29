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

if len(sys.argv) < 2:
  print "Usage python region_network_stats.py dataset_region.csv"

regionDf = pd.read_csv(sys.argv[1])
regionDf['missed_objects'] = regionDf['gt_objects'] - regionDf['detected_objects'] - regionDf['partially_detected_objects']
regionDf.loc[regionDf['missed_objects']<0,'missed_objects'] = 0 

print "Area Covered Stats:"
print "\tMean: " + str(regionDf['area_covered'].mean())
print "\tMax: " + str(regionDf['area_covered'].max())
print "\tMin: " + str(regionDf['area_covered'].min())
print "\tStd Dev: " + str(regionDf['area_covered'].std())

print
print "Missed Objects:"
print "\tTotal: " + str(regionDf['missed_objects'].sum())
print "\tPrct: " + str(float(regionDf['missed_objects'].sum()) / regionDf['gt_objects'].sum())