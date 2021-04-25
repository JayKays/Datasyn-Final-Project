import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import load_model
from config import *

def batch_to_img(xb, idx):
    '''returns a single image given by idx from a batch'''

    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    '''Converts model output to target mask'''

    p = torch.functional.F.softmax(predb[idx], 0)
    return predb[idx].argmax(dim = 0).cpu()

def plot_loss(train_loss, valid_loss):
    '''Plots train and validation loss'''

    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.title(f'Loss of {MODEL_NAME}')
    plt.legend()

def plot_segmentation(xb, yb, predb, num_img = 3):
    '''Visulizes segmentation result'''

    fig, ax = plt.subplots(num_img, 3, figsize=(15,num_img*5))
    fig.suptitle(f'Segmentation results of {MODEL_NAME} on {DATASET}')
    ax[0,0].set_title("Original Image")
    ax[0,1].set_title("Ground Truth")
    ax[0,2].set_title("Prediction")

    for i in range(num_img):
        yb[i,0,0] = 3
        ax[i,0].imshow(batch_to_img(xb,i))
        ax[i,1].imshow(yb[i])
        ax[i,2].imshow(predb_to_mask(predb, i))
    

if __name__ == "__main__":

    baseline_dict = load_model('Baseline')
    final_dict = load_model('Final_model')

    train_loss_b = baseline_dict['train_loss']
    valid_loss_b = baseline_dict['valid_loss']

    train_loss_f = final_dict['train_loss']
    valid_loss_f = final_dict['valid_loss']

    plt.plot(train_loss_b, label='Baseline Train loss')
    plt.plot(valid_loss_b, label='Baseline Valid loss')
    plt.plot(train_loss_f, label='Final Train loss')
    plt.plot(valid_loss_f, label='Final Valid loss')

    plt.title("Loss over training for baseline and final model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()  

        