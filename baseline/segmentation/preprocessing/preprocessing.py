import os
import sys
import argparse
import rasterio
import numpy as np

from os.path import join, basename
#from clearcut_research.preprocessing.image_division import divide_into_pieces
from image_division import divide_into_pieces
#from clearcut_research.preprocessing.binary_mask_converter import poly2mask, split_mask
from binary_mask_converter import poly2mask, split_mask
#from clearcut_research.preprocessing.poly_instances_to_mask import filter_poly
from poly_instances_to_mask import filter_poly

#from clearcut_research.pytorch.utils import get_folders
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
    tiff_path, save_path, land_path, cloud_path,
    width, height,
    polys_path, channels, type_filter,
    filter_by_date, pxl_size_threshold,
    no_merge, pass_chance
):
    print(f'filter_by_date:{filter_by_date}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Save directory created.")
    
    try:
        land_src=rasterio.open(land_path, 'r')
    except IOError:
        print("Land cover map file not found: {}".format(land_path))
        sys.exit()

    for tiff_name in get_folders(tiff_path):
        tiff_filepath = os.path.join(tiff_path, tiff_name)
        
        if no_merge:
            tiff_file = join(save_path, f'{tiff_name}.tif')
        else:
            tiff_file = merge_bands(tiff_filepath, save_path, channels)

        data_path = os.path.join(save_path, basename(tiff_file[:-4]))
        divide_into_pieces(tiff_file, data_path, land_src, width, height)

        mask_pieces_path = os.path.join(data_path, 'masks')
        land_pieces_path = os.path.join(data_path, 'landcover')
        
        clouds_path = os.path.join(cloud_path, basename(tiff_file[:-4])+'_clouds.png')
        if not os.path.exists(clouds_path):
            clouds_pieces_path = None
        else:
            clouds_pieces_path = os.path.join(data_path, 'clouds')
            if not os.path.exists(clouds_pieces_path):
                os.mkdir(clouds_pieces_path)
        
        pieces_info = os.path.join(data_path, 'image_pieces.csv')
        mask_path = poly2mask(
            polys_path, tiff_file, data_path,
            type_filter, filter_by_date
        )
        
        split_mask(mask_path, mask_pieces_path, clouds_path, clouds_pieces_path, pieces_info)

        geojson_polygons = os.path.join(data_path, "geojson_polygons")
        instance_masks_path = os.path.join(data_path, "instance_masks")
        filter_poly(
            poly_pieces_path=geojson_polygons, markup_path=polys_path,
            pieces_info_path=pieces_info, original_image_path=tiff_file,
            image_pieces_path=os.path.join(data_path, 'images'),
            mask_pieces_path=mask_pieces_path,
            land_pieces_path=land_pieces_path,
            clouds_pieces_path=clouds_pieces_path,
            pxl_size_threshold=pxl_size_threshold,
            pass_chance=pass_chance
        )
    land_src.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.'
    )
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons'
    )
    parser.add_argument(
        '--tiff_path', '-tp', dest='tiff_path',
        required=True, help='Path to directory with source tiff folders'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/input',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--land_path', '-ld', dest='land_path',
        default='../data/auxiliary/land.tif',
        help='Path to land cover map file'
    )
    parser.add_argument(
        '--clouds_path', '-cp', dest='clouds_path',
        default=None,
        help='Path to clouds map file'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=224,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'ndvi', 'b8'],
        nargs='+', help='Channels list'
    )
    parser.add_argument(
        '--type_filter', '-tf', dest='type_filter',
        help='Type of clearcut: "open" or "closed")'
    )
    parser.add_argument(
        '--filter_by_date', '-fd', dest='filter_by_date',
        action='store_true', default=False,
        help='Filter by date is enabled'
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
        default=0.3, help='Chance of passing blank tile'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess(
        args.tiff_path, args.save_path, args.land_path, args.clouds_path,
        args.width, args.height,
        args.polys_path, args.channels,
        args.type_filter, args.filter_by_date,
        args.pxl_size_threshold,
        args.no_merge, args.pass_chance
    )