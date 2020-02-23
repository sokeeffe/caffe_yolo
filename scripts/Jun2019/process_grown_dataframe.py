'''
This script reads in a dataframe outputted by eval_grown_region_cover_v2.py
and removes columns that are specific from analysis on the output of 
object_pre_recall.py.
The script outputs a dataframe with only the detections of the grown cells
for various confidence levels
Input Params:
 - dataframe_1.csv
Output Params:
 - dataframe_2.csv

 @author Simon
 @date 25 Jun 2019
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
  print "Usage: python process_grown_dataframe.py dataframe_1.csv dataframe_2.csv"
  exit()

dataframe_1_file = sys.argv[1]
dataframe_2_file = sys.argv[2]
#dataframe_diff_file = sys.argv[3]

df1 = pd.read_csv(dataframe_1_file)

threshold_range = np.arange(0.05,1.0,0.05)
for idx, thresh in enumerate(threshold_range):
  tp_string = 'tp_'+str(thresh)
  fp_string = 'fp_'+str(thresh)
  tn_string = 'tn_'+str(thresh)
  fn_string = 'fn_'+str(thresh)
  del df1[tp_string]
  del df1[fp_string]
  del df1[tn_string]
  del df1[fn_string]

df1.to_csv(dataframe_2_file, index=False)
