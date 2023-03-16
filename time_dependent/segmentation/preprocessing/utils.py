import os
import re
import imageio
import numpy as np


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
        elif ch in ['ndvi', 'b8']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def filter_by_channels(image_tensor, channels):
    result = []
    for ch in channels:
        if ch == 'rgb':
            result.append(image_tensor[:, :, :3])
        elif ch == 'ndvi':
            result.append(image_tensor[:, :, 3:4])
        elif ch == 'b8':
            result.append(image_tensor[:, :, 4:5])
        else:
            raise Exception(f'{ch} channel is unknown!')

    return np.concatenate(result, axis=2)


def get_image_info(instance):
    name_parts = re.split(r'[_.]', instance)
    return '_'.join(name_parts[:2]), '_'.join(name_parts[-3:-1])
