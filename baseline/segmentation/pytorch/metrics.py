import torch
from torch.nn import functional as F


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
