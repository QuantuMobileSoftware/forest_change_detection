import torch
import collections

import numpy as np
import pandas as pd
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, GridDistortion, RandomSizedCrop, RandomBrightnessContrast, Transpose, ElasticTransform, MaskDropout, MedianBlur, CropNonEmptyMaskIfExists)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory
from catalyst.data.sampler import BalanceClassSampler
from utils import get_filepath, read_tensor, filter_by_channels, count_channels

def sampler(df):
    labels = list((df["mask_pxl"]>5)*1)
    return BalanceClassSampler(labels, mode="upsampling")


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
    def __init__(self, channels, dataset_path, image_size, batch_size, num_workers, neighbours, classification_head):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.neighbours = neighbours
        self.classification_head = classification_head

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
            self.channels,
            self.neighbours
        )

        if images_array.ndim == 2:
            images_array = np.expand_dims(images_array, -1)

        masks_array = read_tensor(mask_path)

        aug = Compose([
            RandomRotate90(),
            Flip(),
            OneOf([
                RandomSizedCrop(min_max_height=(int(self.image_size * 0.7), self.image_size),height=self.image_size, width=self.image_size),
                RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                #MedianBlur(blur_limit=3, p=0.2),
                MaskDropout(p=0.6),
                ElasticTransform(alpha=15, sigma=5, alpha_affine=5),
                GridDistortion(p=0.6)
            ], p=0.8),
            ToTensor()
        ])

        augmented = aug(image=images_array, mask=masks_array)
        augmented_images = augmented['image']
        augmented_masks = augmented['mask']
        if self.classification_head:
            masks_class = ((augmented_masks.sum()>0)*1).unsqueeze(-1).float() #.type(torch.FloatTensor)
            return augmented_images, [augmented_masks, masks_class]
        else:
            return {'features': augmented_images, 'targets': augmented_masks}

    def create_loaders(self, train_df, val_df):
        labels = [(x["mask_pxl"]==0)*1 for x in train_df]
        sampler = BalanceClassSampler(labels, mode="upsampling")
        train_loader = UtilsFactory.create_loader(
            train_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler)
        
        labels = [(x["mask_pxl"]==0)*1 for x in val_df]
        sampler = BalanceClassSampler(labels, mode="upsampling")
        valid_loader = UtilsFactory.create_loader(
            val_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler)

        loaders = collections.OrderedDict()
        loaders['train'] = train_loader
        loaders['valid'] = valid_loader

        return loaders
    
    def create_test_loaders(self, test_df):
        test_loader = UtilsFactory.create_loader(
            test_df,
            open_fn=self.get_input_pair,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

        loaders = collections.OrderedDict()
        loaders['test'] = test_df
        return loaders


class SiamDataset(Dataset):
    def __init__(self, num_images, df, phase, channels, dataset_path, image_size, batch_size):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.df = df
        self.num_images = num_images
        self.phase = phase
        self.images_folder = "images"
        self.image_type = "tiff"
        self.masks_folder = "masks"
        self.mask_type = "png"

    def __getitem__(self, idx):
        if len(self.channels) <2 :
            raise Exception('You have to specify at least two channels.')
        
        data_info_row = self.df.iloc[idx]
        instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
        images_array, masks_array = [], []
        for k in range(1,self.num_images+1):
            image_path = get_filepath(
                self.dataset_path, data_info_row['dataset_folder'], self.images_folder,
                instance_name+f'_{k}', file_type=self.image_type
            )

            img = filter_by_channels(
                read_tensor(image_path),
                self.channels, 1
            )
            images_array.append(img)

        
        mask_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.masks_folder,
            instance_name, file_type=self.mask_type
        )
        masks_array=read_tensor(mask_path)

        if self.phase=='train':
            aug = Compose([
                RandomRotate90(),
                Flip(),
                OneOf([
                    RandomSizedCrop(min_max_height=(int(self.image_size * 0.7), self.image_size),height=self.image_size, width=self.image_size),
                    RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                    #MedianBlur(blur_limit=3, p=0.2),
                    MaskDropout(p=0.6),
                    ElasticTransform(alpha=15, sigma=5, alpha_affine=5),
                    GridDistortion(p=0.6)
                ], p=0.8),
                ToTensor()
                ])
        else:
            aug = ToTensor()
        
        '''
        keys = ['image']
        values = [images_array[0]]
        for k in range(self.num_images-1):
            keys.append(f'image{k}')
            values.append(images_array[k+1])
        
        keys.append('mask')
        values.append(masks_array)
        
        #{"image" : images_array[0], "image2" : images_array[1], ..., "mask": masks_array, ...}
        aug_input = { keys[i] : values[i] for i in range(len(keys)) }

        augmented = aug(**aug_input)

        augmented_images = [augmented['image']]
        for k in range(self.num_images-1):
            augmented_images.append(np.transpose(augmented[f'image{k}'], ( 2, 0, 1))/255)

        augmented_masks = [augmented['mask']]

        return {'features': augmented_images, 'targets': augmented_masks, 'name': data_info_row['name'], 'position': data_info_row['position']}
        '''

        augmented = aug(image=np.concatenate( (images_array[0],images_array[1]), axis=-1),mask=masks_array)

        augmented_images = [augmented['image'][:count_channels(self.channels), :,:],
                            augmented['image'][count_channels(self.channels):, :,:]]
        augmented_masks = [augmented['mask']]

        return {'features': augmented_images, 'targets': augmented_masks, 'name': data_info_row['name'], 'position': data_info_row['position']}

    def __len__(self):
        return len(self.df)

class LstmDataset2(Dataset):
    def __init__(self, num_images, df, phase, channels, dataset_path, image_size, batch_size):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.df = df
        self.num_images = num_images
        self.phase = phase
        self.images_folder = "images"
        self.image_type = "tiff"
        self.masks_folder = "masks"
        self.mask_type = "png"

    def __getitem__(self, idx):
        if len(self.channels) <2 :
            raise Exception('You have to specify at least two channels.')
        
        data_info_row = self.df.iloc[idx]
        instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
        
        images_array, masks_array = [], []
        for k in range(1,self.num_images+1):
            image_path = get_filepath(
                self.dataset_path, data_info_row['dataset_folder'], self.images_folder,
                instance_name+f'_{k}', file_type=self.image_type
            )

            img = filter_by_channels(
                read_tensor(image_path),
                self.channels, 1
            )
            images_array.append(img)

        
        mask_path = get_filepath(
            self.dataset_path, data_info_row['dataset_folder'], self.masks_folder,
            instance_name, file_type=self.mask_type
        )
        masks_array.append(read_tensor(mask_path))

        if self.phase=='train':
            aug = ToTensor()
            '''
            aug = Compose([
                RandomRotate90(),
                Flip(),
                OneOf([
                    RandomSizedCrop(min_max_height=(int(self.image_size * 0.7), self.image_size),height=self.image_size, width=self.image_size),
                    RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                    MaskDropout(p=0.6),
                    ElasticTransform(alpha=15, sigma=5, alpha_affine=5),
                    GridDistortion(p=0.6)
                ], p=0.8),
                ToTensor()
                ])
            '''
        else:
            aug = ToTensor()
        
        augmented = aug(image=np.concatenate( images_array, axis=-1),
                        mask=np.concatenate( masks_array, axis=-1) )

        augmented_images = [augmented['image'][num_img*count_channels(self.channels):(num_img+1)*count_channels(self.channels), :,:] for num_img in range(self.num_images)]
        augmented_masks = [augmented['mask']]

        return {'features': augmented_images, 'targets': augmented_masks, 'name': data_info_row['name'], 'position': data_info_row['position']}


    def __len__(self):
        return len(self.df)





class LstmDataset(Dataset):
    def __init__(self, num_images, df, phase, channels, dataset_path, image_size, batch_size, all_masks):
        self.channels = channels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.df = df
        self.num_images = num_images
        self.all_masks = all_masks
        self.phase = phase
        self.images_folder = "images"
        self.image_type = "tiff"
        self.masks_folder = "masks"
        self.mask_type = "png"

    def __getitem__(self, idx):
        if len(self.channels) <2 :
            raise Exception('You have to specify at least two channels.')
        
        data_info_row = self.df.iloc[idx]
        instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
        
        images_array, masks_array = [], []
        #for k in range(1,self.num_images+1):
        for k in range(self.num_images,0,-1):
            image_path = get_filepath(
                self.dataset_path, data_info_row['dataset_folder'], self.images_folder,
                instance_name+f'_{k}', file_type=self.image_type
            )

            img = filter_by_channels(
                read_tensor(image_path),
                self.channels, 1
            )
            images_array.append(img)

        
            mask_path = get_filepath(
                self.dataset_path, data_info_row['dataset_folder'], self.masks_folder,
                instance_name+f'_{k}', file_type=self.mask_type
            )
            msk = read_tensor(mask_path)
            masks_array.append(np.expand_dims(msk, axis=-1) )


        aug = Compose([
            RandomRotate90(),
            Flip(),
            OneOf([
                RandomSizedCrop(min_max_height=(int(self.image_size * 0.7), self.image_size),height=self.image_size, width=self.image_size),
                RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                ElasticTransform(alpha=15, sigma=5, alpha_affine=5),
                GridDistortion(p=0.6)
            ], p=0.8),
            ToTensor()
            ])

        augmented = aug(image=np.concatenate( images_array, axis=-1),
                        mask=np.concatenate( masks_array, axis=-1) )


        augmented_images = torch.stack([augmented['image'][num_img*count_channels(self.channels):(num_img+1)*count_channels(self.channels), :,:] for num_img in range(self.num_images)])
        if self.all_masks:
            augmented_masks = torch.stack([ augmented['mask'][:,:,:,i] for i in range(augmented['mask'].shape[-1]) ] ).squeeze()
        else:
            augmented_masks = torch.stack([augmented['mask'][:,:,:,-1]])

        return {'features': augmented_images, 'targets': augmented_masks, 'name': data_info_row['name'], 'position': data_info_row['position']}


    def __len__(self):
        return len(self.df)




