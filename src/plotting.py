import numpy as np
import torch
from matplotlib import pyplot as plt

def batch_to_img(xb, idx):
    '''returns a single image given by idx from a batch'''

    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    '''Converts model output to target mask'''

    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(dim = 0).cpu()

def plot_loss(train_loss, valid_loss):
    '''Plots train and validation loss'''

    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()

def plot_segmentation(xb, yb, predb, num_img = 3):
    '''Visulizes segmentation result'''

    fig, ax = plt.subplots(num_img, 3, figsize=(15,num_img*5))
    for i in range(num_img):
        ax[i,0].imshow(batch_to_img(xb,i))
        ax[i,1].imshow(yb[i])
        ax[i,2].imshow(predb_to_mask(predb, i))
    
