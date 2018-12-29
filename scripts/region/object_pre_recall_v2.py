'''
This script calculates the precision and recall over a range of confidence levels.
It reads in a CSV file and outputs a precision recall csv file and plots the precision
recall curve. The value precision/recall csv file can then be used later to quickly
plot the precision recall curve
Input Params:
 - detections.csv
Output Params:
 - detections_processed.csv
 - precision_recall.csv
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

if len(sys.argv) != 4:
  print "Usage: python object_pre_recall.py detections.csv processed_detects.csv pre_recall.csv"
  exit()

input_file = sys.argv[1]
processed_output_file = sys.argv[2]
precall_output_file = sys.argv[3]


results_df = pd.read_csv(input_file)
threshold_range = np.arange(0.05,1.0,0.05)

columns = ['thresh','precision','recall']
precall_df = pd.DataFrame(index=np.arange(len(threshold_range)),columns=columns)

gt_overlap = 0.1

for idx, thresh in enumerate(threshold_range):
  tp_string = 'tp_'+str(thresh)
  fp_string = 'fp_'+str(thresh)
  tn_string = 'tn_'+str(thresh)
  fn_string = 'fn_'+str(thresh)

  results_df[tp_string] = 0
  results_df[fp_string] = 0
  results_df[tn_string] = 0
  results_df[fn_string] = 0
  results_df.loc[(results_df['conf']>thresh) & (results_df['gt']>gt_overlap), tp_string] = 1
  results_df.loc[(results_df['conf']>thresh) & (results_df['gt']<gt_overlap), fp_string] = 1
  results_df.loc[(results_df['conf']<thresh) & (results_df['gt']>gt_overlap), fn_string] = 1
  results_df.loc[(results_df['conf']<thresh) & (results_df['gt']<gt_overlap), tn_string] = 1

  tp = results_df[tp_string].sum()
  fp = results_df[fp_string].sum()
  fn = results_df[fn_string].sum()
  tn = results_df[tn_string].sum()

  if (tp+fp) > 0:
    precision = float(tp)/(tp+fp)
  else:
    precision = 0
  if (tp+fn) > 0:
    recall = float(tp)/(tp+fn)
  else:
    recall = 0

  precall_df.loc[idx,'thresh'] = thresh
  precall_df.loc[idx,'precision'] = precision
  precall_df.loc[idx,'recall'] = recall


results_df.to_csv(processed_output_file, index=False)
precall_df.to_csv(precall_output_file)

precall_df = precall_df.sort(['recall'],ascending=[1])
ax_precall = precall_df.plot(x='recall',y='precision',
      xlim=[0.0,1.0],ylim=[0.0,1.0],title='Precision Recall Curve')
ax_precall.set_xlabel("Recall")
ax_precall.set_ylabel("Precision")

plt.show()