import numpy as np
import torch

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice(x,y, smooth = 1e-4):
    intersection = (x * y).sum(1)
    dice = 1 - ((2 * intersection + smooth) / (x.sum(1) + y.sum(1) + smooth))

    return dice

def dice_score(predb, yb):

    num_classes = predb.shape[1]
    batch_size = predb.shape[0]

    batch_scores = []
    total_scores = []
    for c in range(num_classes):
        pred = predb[:,c,:,:].view(batch_size,-1)
        target = yb.view(batch_size,-1)

    return