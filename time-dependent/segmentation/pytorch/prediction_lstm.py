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
from models.u_lstm import ULSTMNet, Unet_LstmDecoder

def predict(
        data_path, model_weights_path, model,
        test_df_path, save_path, size, channels, neighbours
):
    if(model=='lstm_diff'):
        model = ULSTMNet(count_channels(channels), 1, size)
    elif(model=='lstm_decoder'):
        model = Unet_LstmDecoder(count_channels(channels))
    else:
        print('Unknown LSTM model. Return to the default model.')
        model = ULSTMNet(count_channels(channels), 1, size)

    model, device = UtilsFactory.prepare_model(model)
    
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        images_array = []
        for k in range(neighbours,0,-1):
            filename = '_'.join([image_info['name'], image_info['position']])
            image_path = get_filepath(
                data_path, image_info['dataset_folder'],
                'images', filename+f'_{k}',
                file_type='tiff'
            )
            img = filter_by_channels(
                read_tensor(image_path),
                channels, 1
            )
            images_array.append(img)

        
        images = transforms.ToTensor()(np.concatenate( images_array, axis=-1))
        images = torch.stack([images[num_img*count_channels(channels):(num_img+1)*count_channels(channels), :,:] for num_img in range(neighbours)])
        prediction = model.predict(images.unsqueeze(0).to(device, dtype=torch.float))

        result = prediction.view(size, size).detach().cpu().numpy()

        cv.imwrite(
            get_filepath(predictions_path, filename, file_type='png'),
            result * 255
        )
  

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--model', '-n', default='lstm_decoder')
    parser.add_argument('--data_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', required=True, help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', required=True, help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')
    parser.add_argument('--size', '-s', default=56, type=int, help='Image size')
    parser.add_argument('--channels', '-ch', default=['rgb', 'ndvi', 'b8'], nargs='+', help='Channels list')
    parser.add_argument('--neighbours', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(
    args.data_path, args.model_weights_path,
    args.model, args.test_df, args.save_path,
    args.size, args.channels, args.neighbours)
