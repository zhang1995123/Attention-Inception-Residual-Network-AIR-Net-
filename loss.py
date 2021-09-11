import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision


def bce_loss(pred, target):
    criterion = nn.BCELoss()
    pred = F.sigmoid(pred)

    label_smoothing = 0.2
    # target = target * (1-label_smoothing) + label_smoothing / 2

    loss = criterion(pred, target)
    loss = torch.sum(loss)
    return loss

def focal_loss(pred, target, alpha=0.5, gamma=2):
    pred = F.sigmoid(pred)
    # label_smoothing = 0.2
    # target = target * (1-label_smoothing) + label_smoothing / 2
    target = target.view_as(pred)
    BCE_loss = F.binary_cross_entropy(pred, target, reduce=False)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt ** gamma
    loss = focal_weight * BCE_loss
    loss = torch.sum(loss)

    return loss
