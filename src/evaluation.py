import numpy as np
import torch

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb).float().mean()

def dice(x, y, smooth = 1e-4):

    intersection = (x * y).sum(dim = 1)
    dice = (2 * intersection + smooth) / (x.sum(dim = 1) + y.sum(dim = 1) + smooth)

    return dice

def batch_dice(predb, yb):

    num_classes = predb.shape[1]
    batch_size = predb.shape[0]

    # print(num_classes,"\t", batch_size)

    class_scores = []
    total_score = 0
    for c in range(1,num_classes):
        # pred = predb.argmax(dim=1)
        pred = predb[:,c,:,:].view(batch_size,-1)
        target = yb.view(batch_size,-1)

<<<<<<< HEAD
        batch_scores = dice(pred, target)

        class_score = batch_scores.mean()
        class_scores.append(class_score)

        total_score += batch_scores.sum()

    total_score = total_score

    return total_score, class_scores

=======
    return
>>>>>>> 9e1f39638ada79d99e04bb0c54dbc1042df71279
