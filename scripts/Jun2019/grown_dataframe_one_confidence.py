'''
This script reads in a dataframe outputted by eval_grown_region_cover_v2.py
and removes columns from differenct confidence levels so that there is only
one grown output
The script outputs a dataframe with only the detections of the grown cells
at one confidence level
Input Params:
 - dataframe_1.csv
Output Params:
 - dataframe_2.csv

 @author Simon
 @date 26 Jun 2019
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

if len(sys.argv) != 3:
  print "Usage: python grown_dataframe_one_confidence.py dataframe_1.csv dataframe_2.csv"
  exit()

dataframe_1_file = sys.argv[1]
dataframe_2_file = sys.argv[2]

df1 = pd.read_csv(dataframe_1_file)

threshold_range = np.arange(0.1,1.0,0.1)
for idx, thresh in enumerate(threshold_range):
  if thresh == 0.5:
    continue
  grown_string = 'grown_'+str(thresh)
  del df1[grown_string]

df1.to_csv(dataframe_2_file, index=False)
