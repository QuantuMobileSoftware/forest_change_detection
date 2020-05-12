import os
import fiona
import shutil
import imageio
import argparse
import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp

from tqdm import tqdm
from random import random
from rasterio import features
from geopandas import GeoSeries
from itertools import combinations
from shapely.geometry import MultiPolygon


def filter_poly(
    poly_pieces_path, markup_path,
    pieces_info_path, original_image_path,
    image_pieces_path, mask_pieces_path, land_pieces_path, clouds_pieces_path,
    pxl_size_threshold, pass_chance, 
    land_type=40
):
    original_image = rs.open(original_image_path)
    geojson_markup = gp.read_file(markup_path)
    geojson_markup = geojson_markup.to_crs(original_image.crs)

    pieces_info = pd.read_csv(pieces_info_path)

    for i in tqdm(range(len(pieces_info))):
        poly_piece_name = pieces_info['piece_geojson'][i]
        start_x = pieces_info["start_x"][i]
        start_y = pieces_info["start_y"][i]

        x, y = original_image.transform * (start_x + 1, start_y + 1)
        filename, _ = os.path.splitext(poly_piece_name)

        try:
            poly_piece = gp.read_file(os.path.join(poly_pieces_path, poly_piece_name))
        except fiona.errors.DriverError:
            print('Polygon is not found.')
            remove_piece(
                filename, poly_pieces_path,
                image_pieces_path, mask_pieces_path, land_pieces_path
            )
            continue

        intersection = gp.overlay(geojson_markup, poly_piece, how='intersection')
        adjacency_list = compose_adjacency_list(intersection['geometry'])
        components = get_components(intersection['geometry'], adjacency_list)

        multi_polys = []
        for component in components:
            multi_polys.append(MultiPolygon(poly for poly in component))

        mask_piece_file = os.path.join(mask_pieces_path, filename + '.png')
        
        land_piece_file = os.path.join(land_pieces_path, filename + '.png')
        land_piece = imageio.imread(land_piece_file)
        land_piece = (land_piece==land_type).astype(np.uint8)
        
        if clouds_pieces_path:
            cloud_piece_file = os.path.join(clouds_pieces_path, filename + '.png')
            cloud_piece = (imageio.imread(cloud_piece_file )<0.4)*1
        else:
            cloud_piece = np.ones((10,10))

        #Leave all empty and clear images with land_type=40
        if( ((len(multi_polys) == 0 or \
            (imageio.imread(mask_piece_file )).sum() < 255 * pxl_size_threshold) \
        and land_piece.sum()<0.98*land_piece.size) \
        or cloud_piece.sum()<0.3*cloud_piece.size):
            
            remove_piece(
                filename, poly_pieces_path,
                image_pieces_path, mask_pieces_path, land_pieces_path, clouds_pieces_path
            )
        
#        #Remove with probability (1-pass_chance) empty images with land_type=40
#        if random() < pass_chance:
#            continue
#        
#        if(land_piece.sum()>0.98*land_piece.size):
#            
#            remove_piece(
#                filename, poly_pieces_path,
#                image_pieces_path, mask_pieces_path, land_pieces_path, clouds_pieces_path
#            )


def remove_piece(filename, poly_pieces_path, 
                 image_pieces_path, mask_pieces_path, land_pieces_path, cloud_pieces_path):
    geojson_file = os.path.join(poly_pieces_path, filename + '.geojson')
    tiff_file = os.path.join(image_pieces_path, filename + '.tiff')
    mask_file = os.path.join(mask_pieces_path, filename + '.png')
    land_file = os.path.join(land_pieces_path, filename + '.png')
    cloud_file = os.path.join(cloud_pieces_path, filename + '.png')

    if os.path.exists(geojson_file):
        os.remove(geojson_file)
    if os.path.exists(tiff_file):
        os.remove(tiff_file)
    if os.path.exists(mask_file):
        os.remove(mask_file)
    if os.path.exists(land_file):
        os.remove(land_file)
    if os.path.exists(cloud_file):
        os.remove(cloud_file)


def compose_adjacency_list(polys):
    length = len(polys)
    adjacency_list = [set() for x in range(0, length)]
    area_threshold = 20
    for idx_tuple in combinations(range(len(polys)), 2):
        poly1 = polys.iloc[idx_tuple[0]]
        poly2 = polys.iloc[idx_tuple[1]]
        if poly1.intersects(poly2):
            if poly1.buffer(1).intersection(poly2).area > area_threshold:
                adjacency_list[idx_tuple[0]].add(idx_tuple[1])
                adjacency_list[idx_tuple[1]].add(idx_tuple[0])

    return adjacency_list


def bfs(graph, start, visited):
    saved = visited.copy()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

    return visited.difference(saved)


def get_components(polys, adjacency_list):
    visited = set()
    graph_components = []
    for i in range(len(polys)):
        dif = bfs(adjacency_list, i, visited)
        if dif:
            graph_components.append(polys[list(dif)])

    return graph_components


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--geojson_pieces', '-gp', dest='geojson_pieces',
        required=True, help='Path to the directory geojson polygons of image pieces'
    )
    parser.add_argument(
        '--geojson_markup', '-gm', dest='geojson_markup',
        required=True, help='Path to the original geojson markup'
    )
    parser.add_argument(
        '--pieces_info_path', '-pi', dest='pieces_info_path',
        required=True, help='Path to the image pieces info'
    )
    parser.add_argument(
        '--original_image', '-oi', dest='original_image',
        required=True, help='Path to the source tif image'
    )
    parser.add_argument(
        '--image_pieces_path', '-ip', dest='image_pieces_path',
        required=False, help='Image pieces without markup that will be removed'
    )
    parser.add_argument(
        '--mask_pieces_path', '-mp', dest='mask_pieces_path',
        required=False, help='Mask pieces without markup that will be removed'
    )
    parser.add_argument(
        '--pxl_size_threshold', '-mp', dest='pxl_size_threshold',
        default=20, help='Minimum pixel size of mask area'
    )
    parser.add_argument(
        '--pass_chance', '-pc', dest='pass_chance', type=float,
        default=0, help='Chance of passing blank tile'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    markup_to_separate_polygons(
        args.geojson_pieces, args.geojson_markup,
        args.pieces_info_path, args.original_image,
        args.image_pieces_path, args.mask_pieces_path,
        args.pxl_size_threshold
    )
