'''
This script reads in two dataframes stored in CSV format and compares fixed
dataframe columns. The script compares columns of two dataframes specific for 
the post processing step of region_proposal networks. This script should never
be needed again since it was used to successfully confirm functionality of
eval_grown_region_cover_v2.py
Input Params:
 - dataframe_1.csv
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
  print "Usage: python compare_dataframe_columns.py dataframe_1.csv dataframe_2.csv"
  exit()

dataframe_1_file = sys.argv[1]
dataframe_2_file = sys.argv[2]
#dataframe_diff_file = sys.argv[3]

df1 = pd.read_csv(dataframe_1_file)
df2 = pd.read_csv(dataframe_2_file)


if df1['image_name'].equals(df2['image_name']):
  print "image_name data columns are exactly the same"
else:
  print "image_name data columns are different"
if df1['cell_index'].equals(df2['cell_index']):
  print "cell_index data columns are exactly the same"
else:
  print "cell_index data columns are different"
if df1['conf'].equals(df2['conf']):
  print "conf data columns are exactly the same"
else:
  print "conf data columns are different"
if df1['gt'].equals(df2['gt']):
  print "gt data columns are exactly the same"
else:
  print "gt data columns are different"
if df1['grown_0.1'].equals(df2['grown_0.1']):
  print "grown_0.1 data columns are exactly the same"
else:
  print "grown_0.1 data columns are different"
if df1['grown_0.2'].equals(df2['grown_0.2']):
  print "grown_0.2 data columns are exactly the same"
else:
  print "grown_0.2 data columns are different"
if df1['grown_0.3'].equals(df2['grown_0.3']):
  print "grown_0.3 data columns are exactly the same"
else:
  print "grown_0.3 data columns are different"
if df1['grown_0.4'].equals(df2['grown_0.4']):
  print "grown_0.4 data columns are exactly the same"
else:
  print "grown_0.4 data columns are different"
if df1['grown_0.5'].equals(df2['grown_0.5']):
  print "grown_0.5 data columns are exactly the same"
else:
  print "grown_0.5 data columns are different"
if df1['grown_0.6'].equals(df2['grown_0.6']):
  print "grown_0.6 data columns are exactly the same"
else:
  print "grown_0.6 data columns are different"
if df1['grown_0.7'].equals(df2['grown_0.7']):
  print "grown_0.7 data columns are exactly the same"
else:
  print "grown_0.7 data columns are different"
if df1['grown_0.8'].equals(df2['grown_0.8']):
  print "grown_0.8 data columns are exactly the same"
else:
  print "grown_0.8 data columns are different"
if df1['grown_0.9'].equals(df2['grown_0.9']):
  print "grown_0.9 data columns are exactly the same"
else:
  print "grown_0.9 data columns are different"
