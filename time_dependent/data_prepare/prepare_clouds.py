import os
import sys
import cv2
import tifffile
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from os.path import join, splitext, basename, exists

from scipy.signal import medfilt
from s2cloudless import S2PixelCloudDetector
from rasterio.plot import reshape_as_image

def search_band(band, folder, file_type):
    for file in os.listdir(folder):
        if band in file and file.endswith(file_type):
            return splitext(file)[0]
    
    return None


def to_tiff(img_file, output_type='Float32'):
    os.system(
        f'gdal_translate -ot {output_type} \
        {img_file} {splitext(img_file)[0]}.tif'
    )

def merge(save_path, *images):
    os.system(f'gdal_merge.py -ps 40 40 -v -separate -o {save_path} {".tif ".join(images)+".tif"}')


def detect_clouds(merged_tif_path, save_file, scale = 1):
    print('================================')
    print('Cloud detection.')
    cloud_detector = S2PixelCloudDetector()
    
    with rasterio.open(merged_tif_path) as src:
        img = src.read()
        meta = src.meta
    img = reshape_as_image(img)
    img = np.expand_dims(img,axis=0)/10000
    print('predict\t.')
    cloud_probs = cloud_detector.get_cloud_probability_maps(img)
    width = int(img.shape[2] * scale)
    height = int(img.shape[1] * scale)
    print('resize.')
    cloud_probs = cloud_probs.reshape((img.shape[1],img.shape[2]))
    cloud_probs = cv2.resize(cloud_probs, (width, height), 
                             interpolation = cv2.INTER_CUBIC)
    
    print('save cloud.')
    meta['count'] = 1
    meta['height'], meta['width'] = height, width
    meta['dtype'] = cloud_probs.dtype
    with rasterio.open(save_file, 'w', **meta) as dst:
        dst.write(cloud_probs[None,:,:])


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--data_folder', '-f', dest='data_folder',
        required=True, help='Path to downloaded images'
    )
    parser.add_argument(
        '--save_path', '-s', dest='save_path', default='data',
        help='Path to directory where results will be stored'
    )
    return parser.parse_args()

if __name__ == '__main__':    
    args = parse_args()
    granule_folder = join(args.data_folder, 'GRANULE')
    tile_folder = os.listdir(granule_folder)[0]
    if(exists(join(granule_folder, tile_folder, 'IMG_DATA', 'R10m'))): 
        print(f"Level-A product, cloud_probs are already prepared.")
        img_folder = join(granule_folder, tile_folder, 'QI_DATA')
        to_tiff(f'{os.path.join(img_folder,"MSK_CLDPRB_20m.jp2")}')
        clouds_tif_path = os.path.join(img_folder,"MSK_CLDPRB_20m.tif")

        cloud_probs = rasterio.open(clouds_tif_path).read()
        cloud_probs = reshape_as_image(cloud_probs)

        width = int(cloud_probs.shape[0] * 2)
        height = int(cloud_probs.shape[1] * 2)

        cloud_probs = cv2.resize(cloud_probs, (width, height), 
                             interpolation = cv2.INTER_CUBIC)
        cloud_probs = np.clip(cloud_probs, 0, 100)/100

        print('save cloud.')
        save_file_clouds = join(args.save_path, f'{tile_folder}_clouds.tiff')
        imageio.imwrite(save_file_clouds, cloud_probs)
        os.remove(clouds_tif_path)
    else:
        img_folder = join(granule_folder, tile_folder, 'IMG_DATA')
    
        bands, band_names =['B01','B02','B04','B05','B08','B8A','B09','B10','B11','B12'], []

        for band in bands:
            band_names.append(join(img_folder, search_band(band, img_folder, 'jp2')))

        print('\nall bands are converting to *tif...\n')
	    
        for band_name in band_names:
            print(band_name[-3:])
            to_tiff(f'{band_name}.jp2')

        print('\n all bands are being merged...\n')
	    
        save_file_merged = join(args.save_path, f'{tile_folder}_full_merged.tif')
        merge(save_file_merged, *band_names)
	    
        save_file_clouds = join(args.save_path, f'{tile_folder}_clouds.tiff')
        detect_clouds(save_file_merged, save_file_clouds)
        os.remove(save_file_merged)
	    
        for item in os.listdir(img_folder):
            if item.endswith('.tif'):
                os.remove(join(img_folder, item))

        #os.system(f'rm {join(granule_folder, tile_folder, 'IMG_DATA')}*.jp2')
        print('\ntemp files have been deleted\n')