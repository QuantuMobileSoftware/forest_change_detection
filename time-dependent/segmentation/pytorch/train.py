import os
import torch
import argparse
import collections

import numpy as np
import pandas as pd

from catalyst.dl.utils import UtilsFactory
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback, DiceCallback

from torch import nn, cuda
from torch.backends import cudnn

from poutyne.framework import Model
from poutyne.framework.callbacks import ModelCheckpoint
from poutyne.framework.callbacks.lr_scheduler import MultiStepLR

from dataset import Dataset
from utils import count_channels, str2bool
from models.utils import get_model, get_optimizer, get_loss, set_random_seed
from models.metrics import classification_head_accuracy, segmentation_head_dice

import warnings
warnings.simplefilter("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--batch_size', type=int, default=64)
    arg('--num_workers', type=int, default=4)
    arg('--neighbours', type=int, default=1)
    arg('--epochs', '-e', type=int, default=100)
    arg('--lr', type=float, default=1e-3)

    arg('--logdir', default='../logs')
    arg('--train_df', '-td', default='../data/train_df.csv')
    arg('--val_df', '-vd', default='../data/val_df.csv')
    arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')
    arg('--model_weights_path', '-mwp', default=None)
    arg('--name', default='vld_1')
    arg('--optimizer', default='Adam')
    arg('--loss', default='bce_dice')
    arg('--image_size', '-is', type=int, default=224)
    arg('--network', '-n', default='unet50')
    arg('--classification_head', required=True, type=str2bool)
    arg(
        '--channels', '-ch',
        default=['rgb', 'b8', 'b8a', 'b11', 'b12', 'ndvi', 'ndmi'],
        nargs='+', help='Channels list')
    return parser.parse_args()


def train(args):
    set_random_seed(42)
    model = get_model(args.network, args.classification_head)
    print('Loading model')

    model.encoder.conv1 = nn.Conv2d(
        count_channels(args.channels)*args.neighbours, 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False)
    
    model, device = UtilsFactory.prepare_model(model)
    
    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')

    ds = Dataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers, args.neighbours, args.classification_head)
    loaders = ds.create_loaders(train_df, val_df)

    save_path = os.path.join(
        args.logdir,
        args.name
    )

    optimizer = get_optimizer(args.optimizer, args.lr, model)

    if not args.classification_head:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 40, 80, 150, 300], gamma=0.1
        )

        criterion = get_loss(args.loss)
        
        runner = SupervisedRunner()
        if args.model_weights_path:
            checkpoint = torch.load(args.model_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=[
                DiceCallback()
            ],
            logdir=save_path,
            num_epochs=args.epochs,
            verbose=True
        )

        infer_loader = collections.OrderedDict([('infer', loaders['valid'])])
        runner.infer(
            model=model,
            loaders=infer_loader,
            callbacks=[
                CheckpointCallback(resume=f'{save_path}/checkpoints/best.pth'),
                InferCallback()
            ],
        )
    else:
        criterion = get_loss('multi')
        net = Model(model, optimizer, criterion, batch_metrics=[classification_head_accuracy, segmentation_head_dice])
        net = net.to(device)
        net.fit_generator(
            loaders['train'], loaders['valid'],
            epochs=args.epochs,
            callbacks=[ModelCheckpoint(f'{save_path}/checkpoints/best.pth', ), 
                       MultiStepLR(milestones=[10, 40, 80, 150, 300], gamma=0.1)]
            )

if __name__ == '__main__':
    args = parse_args()
    train(args)
