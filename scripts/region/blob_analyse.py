import sys
import os
import Image
import math
import glob
import random
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
  print "Usage python blob_analyse.py blob_data.csv"

blobDf = pd.read_csv(sys.argv[1])
blobDf['aspectRatio'] = blobDf['width']/blobDf['height']
blobDf['normalizedAspectRatio'] = blobDf['aspectRatio']
mask = (blobDf['normalizedAspectRatio']<1)
blobValidDf = blobDf[mask]
blobDf.loc[mask,'normalizedAspectRatio'] = np.reciprocal(blobValidDf['normalizedAspectRatio'])
print "Max: " + str(blobDf['aspectRatio'].max())
print "Min: " + str(blobDf['aspectRatio'].min())

blobDf.to_csv(sys.argv[1], index=False)

print "Unique Values: " + str(blobDf['normalizedAspectRatio'].nunique())

plt.figure()
blobDf['normalizedAspectRatio'].hist(bins=blobDf['normalizedAspectRatio'].nunique())
plt.xlabel('Aspect Ratio Width/Height')
plt.ylabel('Num Blobs')

plt.show()