import os
import random
import argparse
import numpy as np
import pandas as pd
import geopandas as gp

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

#utils import get_filepath, read_tensor, get_folders, get_fullname
from utils import get_filepath, read_tensor, get_folders, get_fullname


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating tables that \
            contain info about all data pathes and names.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data/input', help='Path to the data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--images_folder', '-imf', dest='images_folder',
        default='images',
        help='Name of folder where images are storing'
    )
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--instances_folder', '-inf', dest='instances_folder',
        default='instance_masks',
        help='Name of folder where instances are storing'
    )
    parser.add_argument(
        '--polygons_folder', '-pf', dest='polygons_folder',
        default='geojson_polygons',
        help='Name of folder where polygons are storing'
    )
    parser.add_argument(
        '--image_type', '-imt', dest='image_type',
        default='tiff',
        help='Type of image file'
    )
    parser.add_argument(
        '--mask_type', '-mt', dest='mask_type',
        default='png',
        help='Type of mask file'
    )
    parser.add_argument(
        '--instance_type', '-int', dest='instance_type',
        default='geojson',
        help='Type of instance file'
    )
    parser.add_argument(
        '--split_function', '-sf', dest='split_function',
        default='geo_split',
        help='Available functions: geo_split, stratified_split, '
             + 'train_val_split, fold_split'
    )
    parser.add_argument(
        '--markup_path', '-mp', dest='markup_path',
        required=True,
        help='Path to polygons'
    )
    parser.add_argument(
        '--val_threshold', '-vt', dest='val_threshold',
        default=0.3, type=float,
        help='Split top threshold to specify validation size'
    )
    parser.add_argument(
        '--val_bottom_threshold', '-vbt', dest='val_bottom_threshold',
        default=0.2, type=float,
        help='Split bottom threshold to specify validation size'
    )
    parser.add_argument(
        '--test_threshold', '-tt', dest='test_threshold',
        default=0.2, type=float,
        help='Split threshold to specify test size'
    )
    parser.add_argument(
        '--folds', '-f', dest='folds',
        default=3, type=int,
        help='Folds count'
    )
    return parser.parse_args()


args = parse_args()


def get_instance_info(instance):
    name_parts = os.path.splitext(instance)[0].split('_')
    return '_'.join(name_parts[:-2]), '_'.join(name_parts[-2:])


def add_record(data_info, dataset_folder, name, position):
    return data_info.append(
        pd.DataFrame({
            'dataset_folder': dataset_folder,
            'name': name,
            'position': position
        }, index=[0]),
        sort=True, ignore_index=True
    )


def get_data_info(data_path=args.data_path):
    _, _, insatnces_path = get_data_pathes(data_path)
    instances = get_folders(insatnces_path)
    
    cols = ['dataset_folder', 'name', 'position']
    data_info = pd.DataFrame(columns=cols)
    for instance in instances:
        name, position = get_instance_info(instance)
        data_info = add_record(data_info, dataset_folder=name, name=name, position=position)
        
    return data_info


def get_data_pathes(
    data_path=args.data_path, images_folder=args.images_folder,
    masks_folder=args.masks_folder, instances_folder=args.instances_folder
):
    dataset = get_folders(data_path)[0]
    
    images_path = os.path.join(data_path, dataset, images_folder)
    masks_path = os.path.join(data_path, dataset, masks_folder)
    insatnces_path = os.path.join(data_path, dataset, instances_folder)
    
    return images_path, masks_path, insatnces_path


def stratify(
    data_info, data_path=args.data_path, 
    test_size=0.2, random_state=42,
    instance_type=args.instance_type,
    instances_folder=args.instances_folder
):
    
    X, _ = get_data(data_info)
    areas = []
    for _, row in data_info.iterrows():
        instance_name = get_fullname(row['name'], row['position'])
        instance_path = get_filepath(
            data_path,
            row['name'],
            instances_folder,
            instance_name,
            instance_name,
            file_type=instance_type
        )
        areas.append(get_area(instance_path))
                     
    labels = get_labels(np.array(areas))

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    return sss.split(X, labels)


def get_data(
    data_info, data_path=args.data_path,
    image_folder=args.images_folder, mask_folder=args.masks_folder,
    image_type=args.image_type, mask_type=args.mask_type
):
    
    x = []
    y = []
    for _, row in data_info.iterrows():
        filename = get_fullname(row['name'], row['position'])
        
        image_path = get_filepath(
            data_path,
            row['name'],
            image_folder,
            filename,
            file_type=image_type
        )
        mask_path = get_filepath(
            data_path,
            row['name'],
            mask_folder,
            filename,
            file_type=mask_type
        )
        
        x.append(read_tensor(image_path))
        y.append(read_tensor(mask_path))
        
    x = np.array(x)
    y = np.array(y)
    y = y.reshape([*y.shape, 1])

    return x, y


def get_area(instance_path):
    return (gp.read_file(instance_path)['geometry'].area / 100).median()


def get_labels(distr):
    res = np.full(distr.shape, 3)
    res[distr < np.quantile(distr, 0.75)] = 2
    res[distr < np.quantile(distr, 0.5)] = 1
    res[distr < np.quantile(distr, 0.25)] = 0
    return res


def stratified_split(
    data_info, data_path=args.data_path,
    test_size=0.2, random_state=42,
    instance_type=args.instance_type,
    instances_folder=args.instances_folder
):
    stratified_indexes = stratify(
        data_info, data_path,
        test_size, random_state,
        instance_type,
        instances_folder
    )

    for train_ix, test_ix in stratified_indexes:
        train_df = data_info.iloc[train_ix]
        test_df = data_info.iloc[test_ix]
    
    return train_df, test_df


def get_height_bounds(geometry):
    return geometry.total_bounds[1], geometry.total_bounds[3]


def update_overall_sizes(overall_sizes, test, train, val, deleted):
    overall_sizes["test"] += test
    overall_sizes["train"] += train
    overall_sizes["val"] += val
    overall_sizes["deleted"] += deleted

    return overall_sizes


def geo_split(
    data_path=args.data_path, markup_path=args.markup_path,
    mask_type=args.mask_type, masks_folder=args.masks_folder,
    polygons_folder=args.polygons_folder, test_threshold=0.2,
    val_bottom_threshold=0.2, val_threshold=0.3,
):
    datasets = get_folders(data_path)
    geojson_markup = gp.read_file(markup_path)

    minY, maxY = get_height_bounds(geojson_markup)

    height = maxY - minY

    cols = ['dataset_folder', 'name', 'position']
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)

    overall_sizes = {'test': 0, 'train': 0, 'val': 0, 'deleted': 0}

    for dataset_dir in datasets:
        polys_path = os.path.join(data_path, dataset_dir, polygons_folder)
        print(dataset_dir)

        deleted = 0
        train = 0
        test = 0
        val = 0

        for poly_name in os.listdir(polys_path):
            instance_geojson_path = os.path.join(polys_path, poly_name)
            instance_geojson = gp.read_file(instance_geojson_path)

            if geojson_markup.crs != instance_geojson.crs:
                geojson_markup = geojson_markup.to_crs(instance_geojson.crs)
                minY, maxY = get_height_bounds(geojson_markup)
                height = maxY - minY

            instance_minY, instance_maxY = get_height_bounds(instance_geojson)

            name, position = get_instance_info(poly_name)

            masks_path = os.path.join(data_path, dataset_dir, masks_folder)
            mask_path = get_filepath(
                masks_path,
                get_fullname(name, position),
                file_type=mask_type
            )
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            mask_pixels = np.count_nonzero(mask_array)
            center_pixels = np.count_nonzero(mask_array[10:-10, 10:-10])
            border_pixels = mask_pixels - center_pixels

            if mask_pixels > mask_array.size * 0.001 and center_pixels > border_pixels:
                if instance_maxY < minY + height * test_threshold:
                    test += 1
                    test_df = add_record(test_df, dataset_folder=name, name=name, position=position)
                elif instance_maxY < minY + height * val_threshold \
                        and instance_minY > minY + height * val_bottom_threshold:
                    val += 1
                    val_df = add_record(val_df, dataset_folder=name, name=name, position=position)
                else:
                    train += 1
                    train_df = add_record(train_df, dataset_folder=name, name=name, position=position)
            else:
                deleted += 1

        print("Train size", train, "Validation size", val, "Test size", test)
        print(f"{deleted} images were deleted")
        overall_sizes = update_overall_sizes(overall_sizes, test, train, val, deleted)

    print("Overall sizes", overall_sizes)

    return train_df, val_df, test_df


def fold_split(datasets_path, markup_path, save_path, folds, test_height_threshold=0.3):
    for fold in range(folds):
        train_df, val_df, test_df = geo_split(
            datasets_path, markup_path,
            val_threshold=1 - (1 - test_height_threshold) / folds * fold,
            val_bottom_threshold=1 - (1 - test_height_threshold) / folds * (fold + 1),
            test_threshold=test_height_threshold
        )
        save_split(train_df, f'train{fold}_df', save_path)
        save_split(val_df, f'val{fold}_df', save_path)
        save_split(test_df, f'test{fold}_df', save_path)


def train_val_split(datasets_path, train_val_ratio=0.3):
    random.seed(42)
    datasets = list(os.walk(datasets_path))[0][1]

    cols = ['dataset_folder', 'name', 'position']
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)

    for dataset_dir in datasets:
        polys_path = os.path.join(datasets_path, dataset_dir, "geojson_polygons")
        print(dataset_dir)
        for poly_name in os.listdir(polys_path):
            name, position = get_instance_info(poly_name)
            if random.random() <= train_val_ratio:
                val_df = add_record(val_df, dataset_folder=name, name=name, position=position)
            else:
                train_df = add_record(train_df, dataset_folder=name, name=name, position=position)

    return train_df, val_df


def save_split(split_info, filename, save_path):
    split_info.to_csv(
        get_filepath(save_path, filename, file_type='csv'),
        index=False
    )


if __name__ == '__main__':
    if args.split_function == 'stratified_split':
        data_info = get_data_info()
        train_val_df, test_df = stratified_split(data_info, test_size=args.test_threshold)
        train_df, val_df = stratified_split(train_val_df, test_size=args.val_threshold)

        save_split(train_df, 'train_df', args.save_path)
        save_split(val_df, 'val_df', args.save_path)
        save_split(test_df, 'test_df', args.save_path)
    elif args.split_function == 'geo_split':
        train_df, val_df, test_df = geo_split(
            val_threshold=args.val_threshold,
            val_bottom_threshold=args.val_bottom_threshold,
            test_threshold=args.test_threshold
        )
        save_split(train_df, 'train_df', args.save_path)
        save_split(val_df, 'val_df', args.save_path)
        save_split(test_df, 'test_df', args.save_path)
    elif args.split_function == 'fold_split':
        fold_split(
            args.data_path, args.markup_path,
            args.save_path, args.folds, args.test_threshold
        )
    elif args.split_function == 'train_val_split':
        train_df, val_df = train_val_split(args.data_path, args.val_threshold)

        save_split(train_df, 'train_df', args.save_path)
        save_split(val_df, 'val_df', args.save_path)
    else:
        raise Exception(f'{args.split_function} is an unknown function!')
