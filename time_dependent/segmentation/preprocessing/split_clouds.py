import os
import re
import sys
import imageio
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import io
from os.path import join, basename

from utils import get_folders

import warnings
warnings.filterwarnings('ignore')

def is_image_info(row,masks_pieces_path):
	if os.path.exists('{}/{}.png'.format(
            masks_pieces_path,
            re.split(r'[/.]', row['piece_image'])[-2])):
	    return 1
	else: return 0

def split_cloud(cloud_path, save_cloud_path, image_pieces_path, masks_pieces_path):
    pieces_info = pd.read_csv(
        image_pieces_path, dtype={
            'start_x': np.int64, 'start_y': np.int64,
            'width': np.int64, 'height': np.int64
        }
    )

    pieces_info['is_image'] = pieces_info.apply(lambda row: is_image_info(row,masks_pieces_path), axis=1)
    pieces_info = pieces_info[pieces_info['is_image']==1]
    print('pieces:',pieces_info.shape[0])
    
    clouds = io.imread(cloud_path)
    for i in tqdm(range(pieces_info.shape[0])):
        piece = pieces_info.iloc[i]
        piece_cloud = clouds[
                 piece['start_y']: piece['start_y'] + piece['height'],
                 piece['start_x']: piece['start_x'] + piece['width']
        ]
        filename_cloud = '{}/{}.tiff'.format(
            save_cloud_path,
            re.split(r'[/.]', piece['piece_image'])[-2]
        )
        io.imsave(filename_cloud, piece_cloud)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for dividing cloud maps into pieces.')
    parser.add_argument(
        '--tiff_path', '-tp', dest='tiff_path',
        default='../data/source',
        help='Path to directory with source tiff folders'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/input',
        help='Path to directory where mask will be stored'
    )
    parser.add_argument(
        '--cloud_path', '-cp', dest='cloud_path',
        default='../data/auxiliary',
        help='Path to clouds map file'
    )
    parser.add_argument(
        '--masks_path', '-mp', dest='masks_path',
        default='masks',
        help='Path to masks'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print("Save directory created.")

    for tiff_name in get_folders(args.tiff_path):
        print(tiff_name)
        tiff_filepath = os.path.join(args.tiff_path, tiff_name)
        tiff_file = join(args.save_path, f'{tiff_name}.tif')
        data_path = os.path.join(args.save_path, basename(tiff_file[:-4]))

        clouds_pieces_path = os.path.join(data_path, 'clouds')
        masks_pieces_path = os.path.join(data_path, args.masks_path)
        cloud_path = os.path.join(args.cloud_path, basename(tiff_file[:-4])+'_clouds.tiff')
        pieces_info = os.path.join(data_path, 'image_pieces.csv')
        if not os.path.exists(clouds_pieces_path):
            os.mkdir(clouds_pieces_path)
        if not os.path.exists(masks_pieces_path):
        	print('No masks_pieces_path directory was found')
        	sys.exit()
        split_cloud(cloud_path, clouds_pieces_path, pieces_info, masks_pieces_path)
