import math
import time
import torch
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import tqdm
from torch import nn, cuda
from torch.backends import cudnn
from catalyst.contrib.criterion import LossBinary
from torch.optim.optimizer import Optimizer, required

from .losses import *
from .polyeval import *

def get_satellite_pretrained_resnet(model_weights_path, encoder_name='resnet50'):
    from clearcut_research.pytorch import Autoencoder_Unet
    model = Autoencoder_Unet(encoder_name=encoder_name)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.encoder


def get_model(name, classification_head, model_weights_path=None):
    if name == 'unet34':
        return smp.Unet('resnet34', encoder_weights='imagenet')
    elif name == 'unet18':
        print('classification_head:', classification_head)
        if classification_head:
            aux_params=dict(
                pooling='max',             # one of 'avg', 'max'
                dropout=0.1,               # dropout ratio, default is None
                activation='sigmoid',      # activation function, default is None
                classes=1,                 # define number of output labels
            )
            return smp.Unet('resnet18', aux_params=aux_params, encoder_weights=None, encoder_depth=2, decoder_channels=(256, 128))    
        else:
            return smp.Unet('resnet18', encoder_weights='imagenet', encoder_depth=2, decoder_channels=(256, 128))
    elif name == 'unet50':
        return smp.Unet('resnet50', encoder_weights='imagenet')
    elif name == 'unet101':
        return smp.Unet('resnet101', encoder_weights='imagenet')
    elif name == 'linknet34':
        return smp.Linknet('resnet34', encoder_weights='imagenet')
    elif name == 'linknet50':
        return smp.Linknet('resnet50', encoder_weights='imagenet')
    elif name == 'fpn34':
        return smp.FPN('resnet34', encoder_weights='imagenet')
    elif name == 'fpn50':
        return smp.FPN('resnet50', encoder_weights='imagenet')
    elif name == 'fpn101':
        return smp.FPN('resnet101', encoder_weights='imagenet')
    elif name == 'pspnet34':
        return smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1)
    elif name == 'pspnet50':
        return smp.PSPNet('resnet50', encoder_weights='imagenet', classes=1)
    elif name == 'fpn50_season':
        from clearcut_research.pytorch import FPN_double_output
        return FPN_double_output('resnet50', encoder_weights='imagenet')
    elif name == 'fpn50_satellite':
        fpn_resnet50 = smp.FPN('resnet50', encoder_weights=None)
        fpn_resnet50.encoder = get_satellite_pretrained_resnet(model_weights_path)
        return fpn_resnet50
    elif name == 'fpn50_multiclass':
        return smp.FPN('resnet50', encoder_weights='imagenet', classes=3, activation='softmax')
    else:
        raise ValueError("Unknown network")


def set_random_seed(seed):
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed_all(seed)

    print('Random seed:', seed)

#https://github.com/LiyuanLucasLiu/RAdam
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

def get_optimizer(optimizer_name, lr, model):
    if(optimizer_name=='Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif(optimizer_name=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif(optimizer_name=='RAdam'):
        optimizer = RAdam(model.parameters(), lr=lr)
    else:
        print('Unknown argument. Return to the default optimizer (Adam)')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def get_loss(loss_name):
    if(loss_name=='bce_dice'):
        criterion = BCE_Dice_Loss(bce_weight=0.2)
    elif(loss_name=='bs_bce_dice'):
        criterion = Bootstrapped_BCE_Dice_Loss()
    elif(loss_name=='focal'):
        criterion = FocalLoss()
    elif(loss_name=='cataloss'):
        criterion = LossBinary()
    elif(loss_name=='lovasz'):
        criterion = LovaszHingeLoss()
    elif(loss_name=='tversky'):
        criterion = TverskyLoss()
    elif(loss_name=='multi'):
        print('Loss for multioutput model.')
        criterion = Double_Loss()
    else:
        print('Unknown argument. Return to the default loss (BCE)')
        criterion = BCE_Dice_Loss(bce_weight=0.2)
    return criterion


CUTOFF = 0.5
def dice_coef(y_true, y_pred, base_threshold, eps=1e-7):
    y_pred = (y_pred>base_threshold)*1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps)


class Meter:
    '''A meter to keep track of dice score throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = CUTOFF # <<<<<<<<<<< here's the threshold
        self.dice_scores = []

    def update(self, targets, outputs):
        dice = dice_coef(outputs, targets, self.base_threshold)
        self.dice_scores.append(dice)

    def get_metrics(self):
        dice = np.nanmean(self.dice_scores)
        return dice

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice = meter.get_metrics()
    print("Loss: %0.4f | dice: %0.4f " % (epoch_loss, dice))
    return dice


class Trainer(object):
    def __init__(self, model, lr, batch_size, epochs, criterion, optimizer, scheduler, train_loader, valid_loader, test_loader, save_path):
        self.batch_size = {"train": batch_size, "val": 1}
        self.accumulation_steps = 64 // self.batch_size['train']
        self.lr = lr
        self.num_epochs = epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.net = self.net.to(self.device)
        self.dataloaders = {
            'train': train_loader,
            'val': valid_loader,
            'test': test_loader
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        x, y = [], []
        for img in images:
            #print(img.shape)
            x.append(img.to(self.device, dtype=torch.float))
        for msk in targets:
            #print(msk.shape)
            y.append(msk.to(self.device, dtype=torch.float))

        y = torch.stack(y)
        outputs = self.net(x).squeeze()
        print('mask:',y.shape, 'outputs:',outputs.shape)
        loss = self.criterion(outputs, y.squeeze())
        
        return loss, F.sigmoid(outputs)

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | time of start: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        #for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
        for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
            loss, outputs = self.forward(batch['features'], batch['targets'])
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu().numpy()
            targets =  np.asarray([batch['targets'][i].detach().cpu().numpy() for i in range(len(batch['targets']))])
            meter.update( targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        best_epoch = 0
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("======= New optimal found, saving state =======")
                best_epoch = epoch
                state["best_loss"] = self.best_loss = val_loss
                os.system(f"> {self.save_path}/state.pth")
                torch.save(state, f"{self.save_path}/state.pth")
                torch.save(self.net, f"{self.save_path}/best.pth")
            if epoch - best_epoch > 40:
                print(f'Early stopping. State was saved at epoch {best_epoch}.')
                break
        print('best loss:', self.best_loss)

    def evaluate(self, image_size, channels, dataloader, phase='test'):
        base_threshold = CUTOFF
        model = torch.load(f"{self.save_path}/best.pth")
        model.to(self.device)
        model.eval()
        state = torch.load(f"{self.save_path}/state.pth", map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])

        os.system(f'mkdir {self.save_path}/{phase}')
        predictions = []
        test_polys, truth_polys = [], []
        for i, batch in enumerate(tqdm(dataloader)):
            x, imgs = [], []
            images, masks = batch['features'], batch['targets'][0]
            for img in images:
                x.append(img.to(self.device, dtype=torch.float))
            batch_preds = F.sigmoid(model(x)).squeeze().detach().cpu().numpy()
            masks = masks.squeeze().detach().cpu().numpy()
            for img in x:
                imgs.append(np.transpose(img.squeeze().detach().cpu().numpy(), (1,2,0))[:,:,0:3])

            dice_score = dice_coef(masks, batch_preds, base_threshold, eps=1e-7)

            imageio.imwrite(f"{self.save_path}/{phase}/{batch['name'][0]}_{batch['position'][0]}.png", 
                            ((batch_preds>base_threshold)*255).astype(np.uint8) )
            test_polys.append(polygonize(((batch_preds>base_threshold)*1).astype(np.uint8)))
            truth_polys.append(polygonize(masks.astype(np.uint8)))

            predictions.append([batch['name'][0], batch['position'][0], masks.sum(), ((batch_preds>base_threshold)*1).sum(), dice_score])

            batch_preds = (batch_preds>base_threshold)*1
            '''
            if masks.sum()>0 or batch_preds.sum()>0:
                fig, axs = plt.subplots(1, 4, figsize=(9*4,12))
                axs[0].imshow(imgs[0])
                axs[1].imshow(imgs[1])
                axs[2].imshow(masks.reshape((image_size, image_size)), cmap='binary')
                plt.title(f"{batch['name'][0]}_{batch['position'][0]}")
                axs[3].imshow(batch_preds.reshape((image_size, image_size)), cmap='binary')
                plt.title(f"dice:{round(dice_score,2)}", fontsize=25)
                plt.savefig(f"{self.save_path}/{phase}/{batch['name'][0]}_{batch['position'][0]}")
                plt.close()
            '''

        predictions = pd.DataFrame(predictions)
        predictions.columns = ['name','position', 'msk_pxl', 'pred_pxl', 'dice']
        predictions.to_csv(f'{self.save_path}/{phase}_evaluation.csv')
        print('Dice avg: %.4f'%(predictions['dice'].mean()))
        print('Dice avg is mask: %.4f'%(predictions[predictions['msk_pxl']!=0]['dice'].mean()))
        
        log_save = f'{self.save_path}/{phase}_f1score.csv'
        log = pd.DataFrame(columns=['f1_score','threshold','TP','FP','FN'])
        for threshold in np.arange(0.1, 1, 0.1):
            F1score, true_pos_count, false_pos_count, false_neg_count, total_count = evalfunction(test_polys, truth_polys, threshold=threshold)
            log = log.append({'f1_score': round(F1score,4),
                                'threshold': round(threshold,2),
                                'TP':int(true_pos_count),
                                'FP':int(false_pos_count),
                                'FN':int(false_neg_count)}, ignore_index=True)
        
        print(log)
        log.to_csv(log_save, index=False)






class LstmTrainer(object):
    def __init__(self, model, lr, batch_size, epochs, criterion, optimizer, scheduler, train_loader, valid_loader, test_loader, save_path):
        self.batch_size = {"train": batch_size, "val": 1}
        self.accumulation_steps = 64 // self.batch_size['train']
        self.lr = lr
        self.num_epochs = epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.net = self.net.to(self.device)
        self.dataloaders = {
            'train': train_loader,
            'val': valid_loader,
            'test': test_loader
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        x, y = [], []
        for img in images:
            x.append(img.to(self.device, dtype=torch.float))
        for msk in targets:
            y.append(msk.to(self.device, dtype=torch.float).squeeze())

        y = torch.stack(y)
        if y.ndim==4:
            y=y.permute(1, 0, 2, 3).squeeze()
        outputs = self.net(x)
        outputs = outputs.squeeze()
        #print('y:',y.shape, 'out:', outputs.shape)
        loss = self.criterion(outputs, y)
        
        return loss, F.sigmoid(outputs)

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | time of start: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        #for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
        for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
            loss, outputs = self.forward(batch['features'], batch['targets'])
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu().numpy()
            targets =  np.asarray([batch['targets'][i].detach().cpu().numpy() for i in range(len(batch['targets']))])
            meter.update( targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        best_epoch = 0
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("======= New optimal found, saving state =======")
                best_epoch = epoch
                state["best_loss"] = self.best_loss = val_loss
                os.system(f"> {self.save_path}/state.pth")
                torch.save(state, f"{self.save_path}/state.pth")
                torch.save(self.net, f"{self.save_path}/best.pth")
            if epoch - best_epoch > 40:
                print(f'Early stopping. State was saved at epoch {best_epoch}.')
                break
        print('best loss:', self.best_loss)

    def evaluate(self, image_size, channels, dataloader, phase='test'):
        base_threshold = CUTOFF
        model = torch.load(f"{self.save_path}/best.pth")
        model.to(self.device)
        model.eval()
        state = torch.load(f"{self.save_path}/state.pth", map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])

        os.system(f'mkdir {self.save_path}/{phase}')
        predictions = []
        test_polys, truth_polys = [], []
        for i, batch in enumerate(tqdm(dataloader)):
            x, imgs = [], []
            images, masks = batch['features'], torch.stack(batch['targets'])
            for img in images:
                x.append(img.to(self.device, dtype=torch.float))
            batch_preds = F.sigmoid(model(x)).squeeze().detach().cpu().numpy()
            masks = masks.squeeze().detach().cpu().numpy()
            '''
            for img in x:
                imgs.append(np.transpose(img.squeeze().detach().cpu().numpy(), (1,2,0))[:,:,0:3])

            dice_score = dice_coef(masks, batch_preds, base_threshold, eps=1e-7)

            test_polys.append(polygonize(((batch_preds>base_threshold)*1).astype(np.uint8)))
            truth_polys.append(polygonize(masks.astype(np.uint8)))

            predictions.append([batch['name'][0], batch['position'][0], masks.sum(), ((batch_preds>base_threshold)*1).sum(), dice_score])

            batch_preds = (batch_preds>base_threshold)*1
            '''
            if masks.sum()>0 or batch_preds.sum()>0:
                fig, axs = plt.subplots(2, 5, figsize=(9*5,12))
                for i in range(5):
                    axs[0,i].imshow(masks[i].reshape((image_size, image_size)), cmap='binary')
                    axs[1,i].imshow(batch_preds[i].reshape((image_size, image_size)), cmap='binary')
                plt.show()
            

        predictions = pd.DataFrame(predictions)
        predictions.columns = ['name','position', 'msk_pxl', 'pred_pxl', 'dice']
        predictions.to_csv(f'{self.save_path}/{phase}_evaluation.csv')
        print('Dice avg: %.4f'%(predictions['dice'].mean()))
        print('Dice avg is mask: %.4f'%(predictions[predictions['msk_pxl']!=0]['dice'].mean()))
        
        log_save = f'{self.save_path}/{phase}_f1score.csv'
        log = pd.DataFrame(columns=['f1_score','threshold','TP','FP','FN'])
        for threshold in np.arange(0.1, 1, 0.1):
            F1score, true_pos_count, false_pos_count, false_neg_count, total_count = evalfunction(test_polys, truth_polys, threshold=threshold)
            log = log.append({'f1_score': round(F1score,4),
                                'threshold': round(threshold,2),
                                'TP':int(true_pos_count),
                                'FP':int(false_pos_count),
                                'FN':int(false_neg_count)}, ignore_index=True)
        
        print(log)
        log.to_csv(log_save, index=False)
