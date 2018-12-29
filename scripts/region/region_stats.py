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
  print "Usage python region_stats.py dataset_region.csv"

regionDf = pd.read_csv(sys.argv[1])
no_growth_images = regionDf.loc[regionDf['grown_area_covered'] == 0]['grown_area_covered'].count()
total_images = regionDf['grown_area_covered'].count()
print no_growth_images
print total_images
print float(no_growth_images)/float(total_images)

print regionDf['area_covered'].max()