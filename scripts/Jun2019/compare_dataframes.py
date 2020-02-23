'''
This script reads in two dataframes stored in CSV format and compares them. 
The script compares the two given dataframes and can be used to 
sanity check the functionality of code refactorization or old code whose function
is partially or totally forgotten.
Input Params:
 - dataframe_1.csv
 - dataframe_2.csv

 @author Simon
 @date 23 Jun 2019
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
  print "Usage: python compare_dataframes.py dataframe_1.csv dataframe_2.csv"
  exit()

dataframe_1_file = sys.argv[1]
dataframe_2_file = sys.argv[2]
#dataframe_diff_file = sys.argv[3]

df1 = pd.read_csv(dataframe_1_file)
df2 = pd.read_csv(dataframe_2_file)

## The code commented out below was to do serious. Fancy implementation abandoned
## due to time constraints (poor google searching)
# diff_df = pd.merge(df1, df2, how='outer', indicator='Exist')
# print len(diff_df.index)
# print diff_df['Exist'].value_counts()
# print len(diff_df['Exist']=='both')
# if len(diff_df.index) == len(diff_df['Exist']=='both'):
#   print "Both Dataframes are exactly the same"
# else:
#   print "Dataframes are different, different values are outputed to file"
#   diff_df = diff_df.loc[diff_df['Exist'] != 'both']
#   diff_df.to_csv(dataframe_diff_file)

if df1.equals(df2):
  print "Both Dataframes are exactly the same"
else:
  print "Dataframes are different"