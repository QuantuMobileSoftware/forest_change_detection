import os
import csv
import imageio
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from geopandas import GeoSeries
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.plot import reshape_as_image


def divide_into_pieces(image_path, save_path, width, height):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print('Data directory created.')

    os.makedirs(f'{save_path}/images', exist_ok=True)
    os.makedirs(f'{save_path}/geojson_polygons', exist_ok=True)

    full_mask = imageio.imread(f'{save_path}/full_mask.png')

    with rasterio.open(image_path) as src, open(f'{save_path}/image_pieces.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'original_image', 'piece_image', 'piece_geojson',
            'start_x', 'start_y', 'width', 'height', 'is_mask'
        ])

        for j in tqdm(range(0, src.height // height)):
            for i in range(0, src.width // width):
                raster_window = src.read(
                    window=Window(i * width, j * height, width, height)
                )
                image_array = reshape_as_image(raster_window)[:, :, :3]

                if np.count_nonzero(image_array) > image_array.size * 0.9:
                    filename_w_ext = os.path.basename(image_path)
                    filename, _ = os.path.splitext(filename_w_ext)
                    image_format = 'tiff'
                    piece_name = f'{filename}_{j}_{i}.{image_format}'

                    poly = Polygon([
                        src.xy(j * height, i * width),
                        src.xy(j * height, (i + 1) * width),
                        src.xy((j + 1) * height, (i + 1) * width),
                        src.xy((j + 1) * height, i * width),
                        src.xy(j * height, i * width)
                    ])
                    gs = GeoSeries([poly])
                    gs.crs = src.crs
                    piece_geojson_name = f'{filename}_{j}_{i}.geojson'
                    gs.to_file(
                        f'{save_path}/geojson_polygons/{piece_geojson_name}',
                        driver='GeoJSON'
                    )
                    image_array = reshape_as_image(raster_window)

                    meta = src.meta
                    meta['height'] = image_array.shape[0]
                    meta['width'] = image_array.shape[1]
                    with rasterio.open(f'{save_path}/images/{piece_name}', 'w', **meta) as dst:
                        for ix in range(image_array.shape[2]):
                            dst.write(image_array[:, :, ix], ix + 1)

                        dst.close()

                    is_mask_flag=0
                    if full_mask[j * height: j * height + height,
                                 i * width:  i * width  + width].sum()>0:
                        is_mask_flag=1

                    writer.writerow([
                        filename_w_ext, piece_name, piece_geojson_name,
                        i * width, j * height, width, height, is_mask_flag
                    ])

    csvFile.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for dividing images into smaller pieces.'
    )
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../../data', help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--width', '-w', dest='width',
        default=224, type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height',
        default=224, type=int, help='Height of a piece'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    divide_into_pieces(args.image_path, args.save_path, args.width, args.height)
