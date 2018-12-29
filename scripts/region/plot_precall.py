'''
This script plots the precision recall curves for various networks
The script plots a separate plot for each input file
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

if len(sys.argv) < 2:
  print "Usage: python plot_precall.py network_1_precall.csv network_2_precall.csv ..."
  exit()

precall_files = []
for i in range(len(sys.argv)-1):
  precall_files.append(sys.argv[i+1])

print precall_files

precall_df = pd.read_csv(precall_files[0])

precall_df = precall_df.sort(['recall'],ascending=[1])
ax_precall = precall_df.plot(x='recall',y='precision',
      xlim=[0.0,1.0],ylim=[0.0,1.0],title='Precision Recall Curves')

for i in range(1,len(precall_files)):
  precall_df = pd.read_csv(precall_files[i])
  precall_df = precall_df.sort(['recall'],ascending=[1])
  precall_df.plot(x='recall',y='precision', ax=ax_precall,xlim=[0.0, 1.0], ylim=[0.0, 1.0])

ax_precall.set_xlabel("Recall")
ax_precall.set_ylabel("Precision")
ax_precall.legend(["ShuffleNet B1", "ShuffleNet B2", "ShuffleNet B16", "Darknet Ref B16"], loc=0)

plt.show()