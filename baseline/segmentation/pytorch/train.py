import argparse
import collections
import os

import numpy as np
import pandas as pd
import torch
from catalyst.dl.callbacks import InferCallback, CheckpointCallback, DiceCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory
from torch import nn, cuda
from torch.backends import cudnn

from dataset import Dataset
from losses import BCE_Dice_Loss
from models.utils import get_model
from utils import count_channels


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--batch_size', type=int, default=8)
    arg('--num_workers', type=int, default=4)
    arg('--epochs', '-e', type=int, default=100)
    arg('--lr', type=float, default=1e-3)

    arg('--logdir', default='../logs')
    arg('--train_df', '-td', default='../data/train_df.csv')
    arg('--val_df', '-vd', default='../data/val_df.csv')
    arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')
    arg('--model_weights_path', '-mwp', default='../weights/resnet50-19c8e357.pth')
    arg('--name', default='vld_1')
    arg('--optimizer', default='Adam')
    arg('--image_size', '-is', type=int, default=224)
    arg('--network', '-n', default='unet50')
    arg(
        '--channels', '-ch',
        default=['rgb', 'ndvi', 'b8'],
        nargs='+', help='Channels list')

    return parser.parse_args()


def set_random_seed(seed):
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed_all(seed)

    print('Random seed:', seed)


def train(args):
    set_random_seed(42)
    model = get_model(args.network)
    print('Loading model')
    model.encoder.conv1 = nn.Conv2d(
        count_channels(args.channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False)
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')

    ds = Dataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers)
    loaders = ds.create_loaders(train_df, val_df)

    if(args.optimizer=='Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif(args.optimizer=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        print('Unknown argument. Return to the default optimizer (Adam)')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = BCE_Dice_Loss(bce_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 40], gamma=0.3
    )

    save_path = os.path.join(
        args.logdir,
        args.name
    )

    # model runner
    runner = SupervisedRunner()

    # model training
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


if __name__ == '__main__':
    args = parse_args()
    train(args)
