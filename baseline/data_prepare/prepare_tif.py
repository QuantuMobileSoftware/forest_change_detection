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
            {img_file} {os.path.splitext(img_file)[0]}_scaled.tif'
        )


def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )


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

    b4_name = join(img_folder, search_band('B04', img_folder, 'jp2'))
    b8_name = join(img_folder, search_band('B08', img_folder, 'jp2'))
    rgb_name = join(img_folder, search_band('TCI', img_folder, 'jp2'))
    ndvi_name = join(img_folder, 'ndvi')

    print('\nb4 and b8 bands are converting to *tif...\n')

    to_tiff(f'{b4_name}.jp2')
    to_tiff(f'{b8_name}.jp2')
    to_tiff(f'{rgb_name}.jp2', 'Byte')

    print('\nndvi band is processing...')    

    get_ndvi(f'{b4_name}.tif', f'{b8_name}.tif', f'{ndvi_name}.tif')

    print('\nall bands are scaling to 8-bit images...\n')

    scale_img(f'{ndvi_name}.tif')
    scale_img(f'{b8_name}.jp2')

    print('\nall bands are being merged...\n')

    os.system(
        f'gdal_merge.py -separate -o {save_file} \
        {rgb_name}.tif {ndvi_name}_scaled.tif {b8_name}_scaled.tif'
    )
    
    print('\nsaving in png...\n')

    os.mkdir(png_folder)

    bands = {
        f'{join(png_folder, "rgb.png")}': f'{rgb_name}.tif',
        f'{join(png_folder, "ndvi.png")}': f'{ndvi_name}_scaled.tif',
        f'{join(png_folder, "b8.png")}': f'{b8_name}_scaled.tif'
    }

    for dest, source in tqdm(bands.items()):
        with rasterio.open(source) as src:
            imageio.imwrite(dest, np.moveaxis(src.read(), 0, -1))
            src.close()

    for item in os.listdir(img_folder):
        if item.endswith('.tif'):
            os.remove(join(img_folder, item))

    print('\ntemp files have been deleted\n')