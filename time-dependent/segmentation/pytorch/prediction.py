import os
import torch
import argparse
import collections

import cv2 as cv
import numpy as np
import pandas as pd
import ttach as tta
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import nn
from catalyst.dl.utils import UtilsFactory
from catalyst.dl.experiments import SupervisedRunner


from dataset import Dataset
from models.utils import get_model
from utils import get_filepath, count_channels, read_tensor, filter_by_channels, str2bool

def predict(
        data_path, model_weights_path, network,
        test_df_path, save_path, size, channels, neighbours,
        classification_head
):
    model = get_model(network, classification_head)
    model.encoder.conv1 = nn.Conv2d(
        count_channels(channels)*neighbours, 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )

    model, device = UtilsFactory.prepare_model(model)
    
    if classification_head:
        model.load_state_dict(torch.load(model_weights_path))
    else:
        checkpoint = torch.load(model_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        filename = '_'.join([image_info['name'], image_info['position']])
        image_path = get_filepath(
            data_path, image_info['dataset_folder'],
            'images', filename,
            file_type='tiff'
        )

        image_tensor = filter_by_channels(
            read_tensor(image_path),
            channels,
            neighbours
        )
        if image_tensor.ndim == 2:
            image_tensor = np.expand_dims(image_tensor, -1)

        image = transforms.ToTensor()(image_tensor)
        if classification_head:
            prediction, label = model.predict(image.view(1, count_channels(channels)*neighbours, size, size).to(device, dtype=torch.float))
        else:
            prediction = model.predict(image.view(1, count_channels(channels)*neighbours, size, size).to(device, dtype=torch.float))

        result = prediction.view(size, size).detach().cpu().numpy()

        cv.imwrite(
            get_filepath(predictions_path, filename, file_type='png'),
            result * 255
        )

def temperature_sharping(masks, t=0.5):
    avg = sum([m**t for m in masks])
    return avg/len(masks)

def mean_mask(masks):
    avg = sum([m for m in masks])
    return avg/len(masks)


def tta_pred_eval(
    data_path, model_weights_path, network,
        test_df_path, save_path, size, channels, merge_mode,
         neighbours,classification_head
        ):
    model = get_model(network, classification_head)
    model.encoder.conv1 = nn.Conv2d(
        count_channels(channels)*neighbours, 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )

    model, device = UtilsFactory.prepare_model(model)

    transformers=tta.aliases.d4_transform()    
    
    if classification_head:
        model.load_state_dict(torch.load(model_weights_path))
    else:
        checkpoint = torch.load(model_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        filename = '_'.join([image_info['name'], image_info['position']])
        image_path = get_filepath(
            data_path, image_info['dataset_folder'],
            'images', filename,
            file_type='tiff'
        )

        image_tensor = filter_by_channels(
            read_tensor(image_path),
            channels,
            neighbours
        )
        if image_tensor.ndim == 2:
            image_tensor = np.expand_dims(image_tensor, -1)

        image = transforms.ToTensor()(image_tensor)
        masks = []
        for transformer in transformers:
            augmented_image = transformer.augment_image(image.view(1, count_channels(channels)*neighbours, size, size))
            if classification_head:
                prediction, label = model.predict(augmented_image.to(device, dtype=torch.float))
            else:
                prediction = model.predict(augmented_image.to(device, dtype=torch.float))

            deaug_mask = transformer.deaugment_mask(prediction)
            masks.append(deaug_mask.view(size, size).detach().cpu().numpy())
        
        if merge_mode=='mean':
            result = mean_mask(masks)
        elif merge_mode == 'tsharping':
            result = temperature_sharping(masks)
        else:
            result = mean_mask(masks)
        
        cv.imwrite(
            get_filepath(predictions_path, filename, file_type='png'),
            result * 255
        )
        

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--network', '-n', default='unet50')
    parser.add_argument('--data_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', required=True, help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', required=True, help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')
    parser.add_argument('--size', '-s', default=112, type=int, help='Image size')
    parser.add_argument('--channels', '-ch', default=['rgb', 'ndvi', 'b8'], nargs='+', help='Channels list')
    parser.add_argument('--neighbours', type=int, default=1)
    parser.add_argument('--tta', '-t', default=False, type=bool, help='Use TTA (True | False)')
    parser.add_argument('--merge_mode', '-mm', default='mean', type=str, help='Merge TTA')
    parser.add_argument('--classification_head', required=True, type=str2bool)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.tta:
        tta_pred_eval(
        args.data_path, args.model_weights_path,
        args.network, args.test_df, args.save_path,
        args.size, args.channels, args.merge_mode,
        args.neighbours,args.classification_head
        )
    else:
        predict(
        args.data_path, args.model_weights_path,
        args.network, args.test_df, args.save_path,
        args.size, args.channels, args.neighbours,
        args.classification_head
        )
