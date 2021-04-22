import numpy as np
from utils import *
from matplotlib import pyplot as plt
import torch

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == to_cuda(yb)).float().mean()

def class_dice(predb, yb, smooth = 1e-4):
    '''Return a tensor with average dice score per class over a batch'''

    num_classes = predb.shape[1]
    batch_size = predb.shape[0]

    scores = to_cuda(torch.from_numpy(np.zeros((batch_size, num_classes))))

    for b in range(batch_size):

        pred = predb[b].argmax(dim = 0).view(-1)
        target = yb[b].view(-1)

        for c in range(num_classes):
            class_pred = (pred == c).float()
            class_targ = (target == c).float()

            # print(torch.count_nonzero(class_pred))

            intersection = (class_pred * class_targ).sum() + smooth
            card = class_targ.sum() + class_pred.sum() + smooth

            dice_score = 2*intersection/card

            scores[b,c] = dice_score
    
    class_dices = scores.mean(dim = 0)

    return class_dices

def dice(predb, yb, smooth = 1e-4):
    '''Return average dice score over every class except background'''

    class_score = class_dice(predb, yb)

    return class_score[1:].mean()


            
            


