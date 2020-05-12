#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:20:55 2020

@author: vld-kh
"""

import os
import sys
import imageio
import rasterio
import argparse

import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for plotting the images/masks/lands/clouds.')
    parser.add_argument(
        '--data_path', '-ip', dest='data_path',
        default='../data/input/L1C_T36UYA_A021082_20190706T083605', help='Path to input data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/output',
        help='Path to directory where plots will be stored'
    )
#    parser.add_argument(
#        '-i', dest='i',
#        default=47, type=int,
#        help='Number of row'
#    )
#    parser.add_argument(
#        '-j', dest='j',
#        default=24, type=int,
#        help='Number of column'
#    )

    return parser.parse_args()

def readimg(file, imgtype):
    if(imgtype=='tiff'):
        return reshape_as_image(rasterio.open(file).read())[:,:,:3]
    if(imgtype=='png'):
        return imageio.imread(file)
    
def plotdata(path, i, j, save_path):
    tilename = path.split('/')[-1]
    img = []
    for src,imgtype  in zip(['images','masks','clouds','landcover'],\
                            ['tiff','png','png','png']):
        file = os.path.join(path,src,f'{tilename}_{i}_{j}.{imgtype}')
        if os.path.exists(file):
            img.append(readimg(file, imgtype))
        else:
            img.append(0)
    
    img.append((img[-1]==40)*1)
    titles = ['images','masks','clouds','landcover','grassland']
    if np.sum(img[1])>0:
        fig, ax= plt.subplots(1,len(titles), figsize=(23,9))
        for k in range(5):
            ax[k].set_title(titles[k])
            ax[k].imshow(img[k])
        plt.savefig(f'{save_path}/{tilename}_{i}_{j}.png')
        plt.close()
    
args = parse_args()
if __name__ == '__main__':
    for i in range(50):
        for j in range(50):
            plotdata(args.data_path, i, j, args.save_path)
    
    
    
