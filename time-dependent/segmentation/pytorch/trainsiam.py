import os
import time
import torch
import argparse
import collections

import numpy as np
import pandas as pd

from torch import cuda
from catalyst.dl.utils import UtilsFactory

from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SiamDataset, sampler
from models.siamese3d import Unet3D
from models.siamese import SiamUnet_diff, SiamUnet_conc, Unet

from utils import count_channels, str2bool
from models.utils import get_model, get_optimizer, get_loss, set_random_seed, Trainer

import warnings
warnings.simplefilter("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--model', default='siamdiff')

    arg('--batch_size', type=int, default=64)
    arg('--num_workers', type=int, default=0)
    arg('--neighbours', type=int, default=2)
    arg('--epochs', '-e', type=int, default=100)
    arg('--lr', type=float, default=1e-3)

    arg('--logdir', default='../logs')
    arg('--train_df', '-td', default='../data/train_df.csv')
    arg('--val_df', '-vd', default='../data/val_df.csv')
    arg('--test_df', '-tsd', default='../data/test_df.csv')
    arg('--plot_df', '-pld', default='../data/plot_df.csv')
    arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')
    arg('--model_weights_path', '-mwp', default=None)
    arg('--name', default='vld_1')
    arg('--optimizer', default='Adam')
    arg('--loss', default='bce_dice')
    arg('--image_size', '-is', type=int, default=224)
    arg('--network', '-n', default='unet50')
    arg('--mode', '-md', default='train')
    arg(
        '--channels', '-ch',
        default=['rgb', 'b8', 'b8a', 'b11', 'b12', 'ndvi', 'ndmi'],
        nargs='+', help='Channels list')
    return parser.parse_args()

def train(args):
    set_random_seed(42)
    if(args.model=='unet'):
        model = Unet(count_channels(args.channels)*args.neighbours, 1)
    elif(args.model=='unet3d'):
        model = Unet3D(count_channels(args.channels), 1)
    elif(args.model=='siamdiff'):
        model = SiamUnet_diff(count_channels(args.channels), 1)
    elif(args.model=='siamconc'):
        model = SiamUnet_conc(count_channels(args.channels), 1)
    else:
        print('Unknown siamese model. Return to the default model.')
        model = Unet(count_channels(args.channels)*2, 1)
    if torch.cuda.is_available(): model.cuda()
    print('Loading model')

    model, device = UtilsFactory.prepare_model(model)
    print(device)

    optimizer = get_optimizer(args.optimizer, args.lr, model)
    criterion = get_loss(args.loss)    
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 40, 80, 150, 300], gamma=0.2
    )

    save_path = os.path.join(
        args.logdir,
        args.name
    )
    
    os.system(f"mkdir {save_path}")
    
    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)
    test_df = pd.read_csv(args.test_df)

    train_dataset = SiamDataset(args.neighbours, train_df, 'train',args.channels, args.dataset_path, args.image_size, args.batch_size)
    valid_dataset = SiamDataset(args.neighbours, val_df, 'valid',args.channels, args.dataset_path, args.image_size, args.batch_size)
    test_dataset = SiamDataset(args.neighbours, test_df, 'test',args.channels, args.dataset_path, args.image_size, args.batch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler(train_df))
    valid_loader = DataLoader(valid_dataset, batch_size=1, 
        shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1,
        shuffle=False, num_workers=args.num_workers)

    if args.model_weights_path:
        checkpoint = torch.load(args.model_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # model training    
    model_trainer = Trainer(model, args.lr, args.batch_size, args.epochs, 
                            criterion, optimizer, scheduler,  
                            train_loader, valid_loader, test_loader, 
                            save_path)
    if args.mode=='train':
        model_trainer.start()
    elif args.mode=='eval':
        model_trainer.evaluate(args.image_size, args.channels,
            DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers),
            phase='train')
        model_trainer.evaluate(args.image_size, args.channels,
            DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers),
            phase='val')
        model_trainer.evaluate(args.image_size, args.channels,
            DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers),
            phase='test')
    else:
        print(f'Unknown mode {args.mode}.')

if __name__ == '__main__':
    args = parse_args()
    train(args)
