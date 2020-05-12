import os
import imageio
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from os.path import join, splitext, basename, exists


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


def scale_img(img_file, min_value=0, max_value=255, output_type='Byte'):
    save_file = os.path.splitext(img_file)[0]+'_scaled'
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)

        os.system(
            f'gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {save_file}.tif'
        )
    return save_file



def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )

def get_ndmi(b11_file, b8a_file, ndmi_file):
    os.system(
        f'gdal_calc.py -A {b11_file} -B {b8a_file} \
        --outfile={ndmi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )

def merge(save_path, *images):
    os.system(f'gdal_merge.py -ps 10 10 -v -separate -o {save_path} {".tif ".join(images)+".tif"}')

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
    tile_folder = list(os.walk(granule_folder))[0][1][-1]
    if(exists(join(granule_folder, tile_folder, 'IMG_DATA', 'R10m'))): 
        img_folder = join(granule_folder, tile_folder, 'IMG_DATA', 'R10m')
    else:
        img_folder = join(granule_folder, tile_folder, 'IMG_DATA')
    
    save_file = join(args.save_path, f'{tile_folder}.tif')
    png_folder = join(args.save_path, tile_folder)

    bands, band_names =['TCI','B08','B8A','B10','B11','B12'], []

    for band in bands:
        band_names.append(join(img_folder, search_band(band, img_folder, 'jp2')))
    
    b4_name = join(img_folder, search_band('B04', img_folder, 'jp2'))
    ndvi_name = join(img_folder, 'ndvi')
    ndmi_name = join(img_folder, 'ndmi')
    print('\nall bands are converting to *tif...\n')
    
    for band_name in band_names:
        print(band_name[-3:])
        if 'B08' in band_name: b8_name=band_name
        if 'B8A' in band_name: b8a_name=band_name
        if 'B11' in band_name: b11_name=band_name
        to_tiff(f'{band_name}.jp2')

    to_tiff(f'{b4_name}.jp2')
    print('\nndvi band is processing...')    

    get_ndvi(f'{b4_name}.tif', f'{b8_name}.tif', f'{ndvi_name}.tif')
    
    print('\nndmi band is processing...')    

    get_ndmi(f'{b11_name}.tif', f'{b8a_name}.tif', f'{ndmi_name}.tif')

    band_names.append(ndvi_name)
    band_names.append(ndmi_name)

    bands.append('ndvi')
    bands.append('ndmi')

    print('\nall bands are scaling to 8-bit images...\n')
    band_names_scaled = []
    for band_name in band_names:
        print(band_name)
        scaled_name = scale_img(f'{band_name}.tif')
        band_names_scaled.append(scaled_name)

    print('\nall bands are being merged...\n')
    print(band_names_scaled)
    save_file_merged = join(args.save_path, f'{tile_folder}.tif')
    merge(save_file_merged, *band_names_scaled)
    
    print('\nsaving in png...\n')

    if not os.path.exists(png_folder): 
        os.mkdir(png_folder)

    for i in range(len(band_names_scaled)):
        print(bands[i], band_names_scaled[i]+'.tif')
        dest, source = f'{join(png_folder, bands[i]+".png")}', band_names_scaled[i]+'.tif'
        with rasterio.open(source) as src:
            imageio.imwrite(dest, np.moveaxis(src.read(), 0, -1))
            src.close()

    for item in os.listdir(img_folder):
        if item.endswith('.tif'):
            os.remove(join(img_folder, item))

    print('\ntemp files have been deleted\n')