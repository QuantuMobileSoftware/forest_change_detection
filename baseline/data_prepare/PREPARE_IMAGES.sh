#!/bin/bash

datafolder=$1  #EXAMPLE: /home/USER/path/TRAINING_SET/IMGS/
savepath=$2    #EXAMPLE: /home/USER/path/TRAINING_SET/data/

for dir in $datafolder/*/; 
do
	echo "$dir"
	python prepare_tif.py --data_folder $dir --save_path $savepath
#	python prepare_clouds.py --data_folder $dir --save_path $savepath --prob 0.6
done