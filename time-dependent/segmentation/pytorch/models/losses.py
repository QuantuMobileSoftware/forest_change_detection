import torch
from catalyst.dl.metrics import dice
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable

from .metrics import multi_class_dice

ALPHA = 0.5
BETA = 1-ALPHA #0.1

class TverskyLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

ALPHA = 0.8
GAMMA = 2
class FocalLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class BCE_Dice_Loss(Module):
    def __init__(self, bce_weight=0.5):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, x, y):
        bce = F.binary_cross_entropy_with_logits(x, y)

        dice = dice_loss(x, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        
        return loss


class Double_Loss(Module):
    def __init__(self, bce_weight=0.1, dice_weight=0.6, class_weigh=0.3):
        super(Double_Loss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.class_weigh = class_weigh

    def forward(self, x, y):
        x_segm, x_class = x[0], x[1]
        y_segm, y_class = y[0], y[1]
        
        bce = F.binary_cross_entropy_with_logits(x_segm, y_segm)
        dice = dice_loss(x_segm, y_segm)

        bce_class = F.binary_cross_entropy_with_logits(x_class, y_class)

        loss = bce * self.bce_weight + dice * self.dice_weight
        loss += bce_class * self.class_weigh
        return loss


class Bootstrapped_BCE_Dice_Loss(Module):
    def __init__(self, bce_weight=0.5):
        super(Bootstrapped_BCE_Dice_Loss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, x, y):
        sbl = SoftBootstrappingLoss()

        bce_bootstrapped = sbl.forward(x, y)

        dice = dice_loss(x, y)

        loss = bce_bootstrapped * self.bce_weight + dice * (1 - self.bce_weight)

        return loss


class SoftBootstrappingLoss(Module):
    def __init__(self, beta=0.95, reduce=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, x, y):
        prediction_tensor = x
        target_tensor = y

        bootstrap_target_tensor = self.beta * target_tensor + (1.0 - self.beta) * torch.sigmoid(prediction_tensor)

        return F.binary_cross_entropy_with_logits(input=prediction_tensor, target=bootstrap_target_tensor)


def dice_loss(pred, target, eps=1.):
    return 1 - dice(pred, target, eps)


class MultiClass_Dice_Loss(Module):
    def __init__(self, ce_weight=0.2):
        super(MultiClass_Dice_Loss, self).__init__()
        self.ce_weight = ce_weight

    def forward(self, x, y):
        ce = CrossEntropyLoss()(x, y)

        dice = multi_class_dice_loss(x, y)

        loss = ce * self.ce_weight + dice * (1 - self.ce_weight)

        return loss


def multi_class_dice_loss(input, target):
    return 1 - multi_class_dice(input, target)


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

class LovaszHingeLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz
