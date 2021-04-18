import numpy as np
import torch
from matplotlib import pyplot as plt

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()


def plot_loss(train_loss, valid_loss):
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    # plt.show()

def plot_visual_results(bs, xb, yb, predb):
    fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
    for i in range(bs):
        ax[i,0].imshow(batch_to_img(xb,i))
        ax[i,1].imshow(yb[i])
        ax[i,2].imshow(predb_to_mask(predb, i))
    
