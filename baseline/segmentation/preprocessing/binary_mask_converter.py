import os
import re
import imageio
import datetime
import argparse
import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp

from rasterio import features


def poly2mask(
    polys_path, image_path, save_path,
    type_filter=None, filter_by_date=True
):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Output directory created.")

    markup = gp.read_file(polys_path)

    if filter_by_date:
        markup['img_date'] = markup['img_date'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        )

        original_image_filename = os.path.basename(os.path.normpath(image_path))
        dt = original_image_filename.split('_')[3][0:8]
        if dt.endswith('T'):
            dt = datetime.datetime.strptime(dt[:-1], '%Y%m%d')
        else:
            dt = datetime.datetime.strptime(dt, '%Y%m%d')

        if markup[markup['img_date'] <= dt].size == 0:
            dt = markup['img_date'][0]

        dt += datetime.timedelta(days=1)
        polys = markup[markup['img_date'] <= dt].loc[:, 'geometry']
    else:
        polys = markup.loc[:, 'geometry']
    
    print('Number of polygons:',len(polys))
    with rs.open(image_path) as image:
        polys = polys.to_crs({'init': image.crs})

        if type_filter in ['open', 'overgrown']:
            polys = polys[markup['state'] == type_filter]
        elif type_filter is not None:
            raise Exception(f'{type_filter} is unknown type! "open" or "overgrown" is available.')

        mask = features.rasterize(
            shapes=polys,
            out_shape=(image.height, image.width),
            transform=image.transform,
            default_value=255
        )

    image.close()

    filename = '{}/{}.png'.format(
        save_path,
        re.split(r'[/.]', image_path)[-2]
    )
    imageio.imwrite(filename, mask)

    return filename


def split_mask(mask_path, save_mask_path, cloud_path, save_cloud_path, image_pieces_path):
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    pieces_info = pd.read_csv(
        image_pieces_path, dtype={
            'start_x': np.int64, 'start_y': np.int64,
            'width': np.int64, 'height': np.int64
        }
    )
    mask = imageio.imread(mask_path)
    if save_cloud_path:
        clouds = imageio.imread(cloud_path)
    for i in range(pieces_info.shape[0]):
        piece = pieces_info.loc[i]
        piece_mask = mask[
             piece['start_y']: piece['start_y'] + piece['height'],
             piece['start_x']: piece['start_x'] + piece['width']
        ]
        filename_mask = '{}/{}.png'.format(
            save_mask_path,
            re.split(r'[/.]', piece['piece_image'])[-2]
        )
        if save_cloud_path:
            piece_cloud = clouds[
                 piece['start_y']: piece['start_y'] + piece['height'],
                 piece['start_x']: piece['start_x'] + piece['width']
            ]
            filename_cloud = '{}/{}.png'.format(
                save_cloud_path,
                re.split(r'[/.]', piece['piece_image'])[-2]
            )
            imageio.imwrite(filename_cloud, piece_cloud)
            imageio.imwrite(filename_mask, np.multiply(piece_mask/255, piece_cloud/255)*255)
        else:
            imageio.imwrite(filename_mask, piece_mask)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons'
    )
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../output',
        help='Path to directory where mask will be stored'
    )
    parser.add_argument(
        '--mask_path', '-mp', dest='mask_path',
        help='Path to the mask',
        required=False
    )
    parser.add_argument(
        '--pieces_path', '-pcp', dest='pieces_path',
        help='Path to directory where pieces will be stored',
        default='../output/masks'
    )
    parser.add_argument(
        '--pieces_info', '-pci', dest='pieces_info',
        help='Path to the image pieces info'
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mask_path = poly2mask(
        args.polys_path, args.image_path,
        args.save_path, args.type_filter
    )
    if args.mask_path is not None:
        mask_path = args.mask_path

    split_mask(mask_path, args.pieces_path, args.pieces_info)
