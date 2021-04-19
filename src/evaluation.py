import numpy as np
from utils import *
from matplotlib import pyplot as plt
import torch

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == to_cuda(yb)).float().mean()

# def dice(x, y, smooth = 1e-4):
    
#     intersection = (x * y).sum(dim = 1)
#     dice = (2 * intersection + smooth) / (x.sum(dim = 1) + y.sum(dim = 1) + smooth)

#     return dice

# def batch_dice(predb, yb):

#     num_classes = predb.shape[1]
#     batch_size = predb.shape[0]

#     # print(num_classes,"\t", batch_size)

#     class_scores = []
#     total_score = 0
#     for c in range(1,num_classes):
#         # pred = predb.argmax(dim=1)
#         # x = predb[0,c,:,:]
#         # print(x.shape)
#         # x = x.cpu()
#         # plt.imshow(x.detach().numpy())
#         # plt.colorbar()
#         # plt.show()

#         pred = predb[:,c,:,:].view(batch_size,-1)
#         target = yb.view(batch_size,-1)

#         batch_scores = dice(pred, target)

#         class_score = batch_scores.mean()
#         class_scores.append(class_score)

#         total_score += batch_scores.sum()

#     total_score = total_score

#     return total_score, class_scores


def dice(predb, yb, smooth = 1e-4):

    num_classes = predb.shape[1]
    batch_size = predb.shape[0]

    scores = to_cuda(torch.from_numpy(np.zeros((batch_size, num_classes))))

    for b in range(batch_size):

        pred = predb[b].argmax(dim = 0).view(-1)
        target = yb[b].view(-1)

        for c in range(1,num_classes):
            class_pred = (pred == c).float()
            class_targ = (target == c).float()

            # print(torch.count_nonzero(class_pred))

            intersection = (class_pred * class_targ).sum() + smooth
            card = class_targ.sum() + class_pred.sum() + smooth

            # print("intersection = ", intersection)
            # print("Card = ", card)

            dice_score = 2*intersection/card

            scores[b,c] = dice_score
    
    class_dices = scores.mean(dim = 1)
    tot_dice = scores.mean()

    return tot_dice, class_dices

def dice3(predb, yb):
    return 
    


            
            


