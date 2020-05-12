import torch
from torch.nn import functional as F
from catalyst.dl.metrics import dice

def multi_class_dice(input: torch.Tensor, target: torch.Tensor, eps=1e-7,
                     threshold: float = None,
                     activation: str = "Softmax"):
    num_classes = input.shape[1]
    true_1_hot = torch.eye(num_classes)[target]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(input, dim=1)
    true_1_hot = true_1_hot.type(input.type())
    dims = (0,) + tuple(range(2, input.ndimension()))

    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    open_dice = ((2. * intersection[1] + eps) / (cardinality[1] + eps)).mean()
    overgrown_dice = ((2. * intersection[2] + eps) / (cardinality[2] + eps)).mean()
    return (open_dice + overgrown_dice) / 2


def classification_head_accuracy(y_pred, y_true):
    """
    Computes the accuracy for a batch of predictions
    
    Args:
        y_pred (torch.Tensor): the logit predictions of the neural network.
        y_true (torch.Tensor): the ground truths.
        
    Returns:
        The average accuracy of the batch.
    """
    y_pred = y_pred[1]
    y_true = y_true[1]
    y_pred = (y_pred>0.5)*1
    return (y_pred == y_true).float().mean()

def segmentation_head_dice(y_pred, y_true):
    """
    Computes the accuracy for a batch of predictions
    
    Args:
        y_pred (torch.Tensor): the logit predictions of the neural network.
        y_true (torch.Tensor): the ground truths.
        
    Returns:
        The average accuracy of the batch.
    """
    y_pred = y_pred[0]
    y_true = y_true[0]
    return dice(y_pred, y_true, 1e-7)