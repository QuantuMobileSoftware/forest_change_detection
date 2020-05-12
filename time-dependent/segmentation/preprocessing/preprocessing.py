import os
import argparse
import rasterio
import numpy as np

from os.path import join, basename
from image_division import divide_into_pieces
from poly_instances_to_mask import filter_poly

from utils import get_folders

def scale_img(img_file, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)
        
        os.system(
            f"gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {f'{os.path.splitext(img_file)[0]}_scaled.tif'}"
        )


def get_ndvi(data_folder, save_path):
    b4_file = join(data_folder, f'{basename(data_folder)}_b4.tif')
    b8_file = join(data_folder, f'{basename(data_folder)}_b8.tif')
    ndvi_file = join(data_folder, f'{basename(data_folder)}_ndvi.tif')

    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )

    return ndvi_file


def merge(save_path, *images):
    os.system(f'gdal_merge.py -separate -o {save_path} {" ".join(images)}')


def merge_bands(tiff_filepath, save_path, channels):
    for file in os.listdir(tiff_filepath):
        if file.endswith('.tif'):
            tiff_file = file
            break

    image_name = '_'.join(tiff_file.split('_')[:2])
    image_path = os.path.join(save_path, f'{image_name}.tif') 
    file_list = []

    for i, channel in enumerate(channels):
        img = os.path.join(tiff_filepath, '_'.join([image_name, channel]))
        if channel == 'rgb':
            file_list.append(f'{img}.tif')
        elif channel == 'ndvi':
            scale_img(get_ndvi(tiff_filepath, tiff_filepath))
            file_list.append(f'{img}_scaled.tif')
        else:
            scale_img(f'{img}.tif')
            file_list.append(f'{img}_scaled.tif')

    merge(image_path, *file_list)

    return image_path


def preprocess(
    tiff_path, tiff_name, save_path,
    width, height,
    polys_path, channels, type_filter,
    pxl_size_threshold,
    no_merge, pass_chance
):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Save directory created.")

    #for tiff_name in get_folders(tiff_path):
    tiff_filepath = os.path.join(tiff_path, tiff_name)
    if no_merge:
        tiff_file = join(save_path, f'{tiff_name}.tif')
    else:
        tiff_file = merge_bands(tiff_filepath, save_path, channels)
    
    data_path = os.path.join(save_path, basename(tiff_file[:-4]))

    # Full mask
    poly2mask(
        polys_path, tiff_file, data_path,
        type_filter, filter_by_date=False
    )
    # Time-series mask
    mask_path, markup = poly2mask(
        polys_path, tiff_file, data_path,
        type_filter, filter_by_date=True
    )

    divide_into_pieces(tiff_file, data_path, width, height)
    
    mask_pieces_path = os.path.join(data_path, 'masks')
    pieces_info = os.path.join(data_path, 'image_pieces.csv')

    split_mask(mask_path, mask_pieces_path, pieces_info)

    geojson_polygons = os.path.join(data_path, "geojson_polygons")
    instance_masks_path = os.path.join(data_path, "instance_masks")
    filter_poly(
        poly_pieces_path=geojson_polygons, markup_path=polys_path,
        pieces_info_path=pieces_info, original_image_path=tiff_file,
        image_pieces_path=os.path.join(data_path, 'images'),
        mask_pieces_path=mask_pieces_path, 
        pxl_size_threshold=pxl_size_threshold,
        pass_chance=pass_chance,
        markup=markup
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.'
    )
    parser.add_argument(
        '--tiff_file', '-tf', dest='tiff_file',
        required=True, help='Name of the directory with source tiff folders'
    )
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        default='../data/polygons', help='Path to the polygons'
    )
    parser.add_argument(
        '--tiff_path', '-tp', dest='tiff_path',
        default='../data/source', help='Path to directory with source tiff folders'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/input',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=56,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=56,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'ndvi', 'b8'],
        nargs='+', help='Channels list'
    )
    parser.add_argument(
        '--type_filter', '-tc', dest='type_filter',
        help='Type of clearcut: "open" or "closed")'
    )
    parser.add_argument(
        '--pxl_size_threshold', '-mp', dest='pxl_size_threshold',
        default=20, help='Minimum pixel size of mask area'
    )
    parser.add_argument(
        '--no_merge', '-nm', dest='no_merge',
        action='store_true', default=False,
        help='Skip merging bands'
    )
    parser.add_argument(
        '--pass_chance', '-pc', dest='pass_chance', type=float,
        default=0, help='Chance of passing blank tile'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess(
        args.tiff_path, args.tiff_file, args.save_path,
        args.width, args.height,
        args.polys_path, args.channels,
        args.type_filter,
        args.pxl_size_threshold,
        args.no_merge, args.pass_chance
    )