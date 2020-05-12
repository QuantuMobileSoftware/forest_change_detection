#!/bin/bash

datafolder=$1  #EXAMPLE: ../data/source

for dir in $datafolder/*; 
do
	dir=`echo "$dir" | sed "s!$datafolder/!!g"`
	echo "$dir"
	python preprocessing.py --tiff_file $dir --no_merge
done

python split_clouds.py --cloud_path ../data/auxiliary