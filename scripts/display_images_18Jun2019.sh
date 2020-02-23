#!/bin/bash

# display_images_18Jun2019.sh
# Script calls display_LMDB_image.sh for every csv image and labe file outputed
# when training a network in a debug mode.

# @author: 	Simon
# @date:	18Jun2019

for batch in {0..2}
do
  for image in {0..31}
  do
    ./scripts/display_LMDB_image.sh models_June19/debug_RP9/lmdb_input_"$batch"_"$image"_3_288_288.csv models_June19/debug_RP9/modified_labels_"$batch"_"$image"_3_288_288.csv
  done
done 
