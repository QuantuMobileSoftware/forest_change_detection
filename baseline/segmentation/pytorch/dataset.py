import collections

import numpy as np
import pandas as pd
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomSizedCrop)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory

from clearcut_research.pytorch.utils import get_filepath, read_tensor, filter_by_channels


def add_record(data_info, dataset_folder, name, position):
    return data_info.append(
        pd.DataFrame({
            'dataset_folder': dataset_folder,
            'name': name,
            'position': position
        }, index=[0]),
        sort=True, ignore_index=True
    )


class Dataset:
    def __init__(self, channels, dataset_path, image_size, batch_size, num_workers):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.images_folder = "images"
        self.image_type = "tiff"
        self.masks_folder = "masks"
        self.mask_type = "png"

    def get_input_pair(self, data_info_row):
        if len(self.channels) == 0:
            raise Exception('You have to specify at least one channel.')

        instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
        image_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.images_folder,
            instance_name, file_type=self.image_type
        )
        mask_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.masks_folder,
            instance_name, file_type=self.mask_type
        )

        images_array = filter_by_channels(
            read_tensor(image_path),
            self.channels
        )

        if images_array.ndim == 2:
            images_array = np.expand_dims(images_array, -1)

        masks_array = read_tensor(mask_path)

        if self.channels[0] == 'rgb':
            rgb_tensor = images_array[:, :, :3].astype(np.uint8)

            rgb_aug = Compose([
                OneOf([
                    RGBShift(),
                    CLAHE(clip_limit=2)
                ], p=0.4)
            ], p=0.9)

            augmented_rgb = rgb_aug(image=rgb_tensor, mask=masks_array)
            images_array = np.concatenate([augmented_rgb['image'], images_array[:, :, 3:]], axis=2)
            masks_array = augmented_rgb['mask']

        aug = Compose([
            RandomRotate90(),
            Flip(),
            OneOf([
                RandomSizedCrop(
                    min_max_height=(int(self.image_size * 0.7), self.image_size),
                    height=self.image_size, width=self.image_size)
            ], p=0.4),
            ToTensor()
        ])

        augmented = aug(image=images_array, mask=masks_array)
        augmented_images = augmented['image']
        augmented_masks = augmented['mask']

        return {'features': augmented_images, 'targets': augmented_masks}

    def create_loaders(self, train_df, val_df):
        train_loader = UtilsFactory.create_loader(
            train_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

        valid_loader = UtilsFactory.create_loader(
            val_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

        loaders = collections.OrderedDict()
        loaders['train'] = train_loader
        loaders['valid'] = valid_loader

        return loaders
