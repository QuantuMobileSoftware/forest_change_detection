import os
import re
import imageio
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return f'{join_pathes(*path_parts)}.{file_type}'


def read_tensor(filepath):
    return imageio.imread(filepath)


def get_folders(path):
    return list(os.walk(path))[0][1]


def split_fullname(fullname):
    return fullname.split('_')


def get_fullname(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        else:
            count += 1
    return count


def filter_by_channels(image_tensor, channels, neighbours):
    #['TCI','B08','B8A','B10','B11','B12', 'NDVI', 'NDMI']
    result = []
    for i in range(neighbours):
        for ch in channels:
            if ch == 'rgb':
                result.append(image_tensor[:, :, (0+i*10):(3+i*10)])
            elif ch == 'b8':
                result.append(image_tensor[:, :, (3+i*10):(4+i*10)])
            elif ch == 'b8a':
                result.append(image_tensor[:, :, (4+i*10):(5+i*10)])
            elif ch == 'b10':
                result.append(image_tensor[:, :, (5+i*10):(6+i*10)])
            elif ch == 'b11':
                result.append(image_tensor[:, :, (6+i*10):(7+i*10)])
            elif ch == 'b12':
                result.append(image_tensor[:, :, (7+i*10):(8+i*10)])
            elif ch == 'ndvi':
                result.append(image_tensor[:, :, (8+i*10):(9+i*10)])
            elif ch == 'ndmi':
                result.append(image_tensor[:, :, (9+i*10):(10+i*10)])
            else:
                raise Exception(f'{ch} channel is unknown!')
    return np.concatenate(result, axis=2)


def get_image_info(instance):
    name_parts = re.split(r'[_.]', instance)
    return '_'.join(name_parts[:2]), '_'.join(name_parts[-3:-1])
