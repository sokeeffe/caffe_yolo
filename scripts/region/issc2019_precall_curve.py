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

if len(sys.argv) != 2:
  print "Usage: python issc2019_precall_curves.py pre_recall.csv"
  exit()

precall1_file = sys.argv[1]
# precall2_file = sys.argv[2]


precall1_df = pd.read_csv(precall1_file)
# precall2_df = pd.read_csv(precall2_file)
# threshold_range = np.arange(0.05,1.0,0.05)

# columns = ['thresh','precision','recall']
# precall_df = pd.DataFrame(index=np.arange(len(threshold_range)),columns=columns)

# gt_overlap = 0.1



# results_df.to_csv(processed_output_file, index=False)
# precall_df.to_csv(precall_output_file)

precall1_df = precall1_df.sort(['recall'],ascending=[1])
# precall2_df = precall2_df.sort(['recall'],ascending=[1])
ax_precall = precall1_df.plot(x='recall',y='precision',
      xlim=[0.75,1.0],ylim=[0.75,1.0],title='Precision Recall Curve')
# precall2_df.plot(ax=ax_precall, x='recall', y='precision',
#   xlim=[0.75,1.0],ylim=[0.75,1.0],title='Precision Recall Curve')
ax_precall.set_xlabel("Recall")
ax_precall.set_ylabel("Precision")
ax_precall.legend(['Region Proposal 9'], loc=3)

plt.show()