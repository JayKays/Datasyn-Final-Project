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
    

def plot_bar_chart():
    '''Makes bar chart over different model changes'''

    data_avg = {'baseline':0.8499, '5_layers':0.8856, 'contrast':0.8853,
        'gaussian':0.8857, 'SGD': 0.8793, 'sharpen': 0.8886, 'final_model': 0.8939}
    data_LV = {'baseline':0.8946, '5_layers':0.9129, 'contrast':0.9115,
        'gaussian':0.9158, 'SGD': 0.9094, 'sharpen': 0.9169, 'final_model': 0.9181}

    labels = list(data_LV.keys())
    values_LV = list(data_LV.values())
    values_avg = list(data_avg.values())

    bar_width = 0.33
    br1 = np.arange(len(labels))
    br2 = [x + bar_width for x in br1]

    fig = plt.figure(figsize =(12, 8))
    plt.bar(br1, values_avg, width = bar_width, label = 'average', zorder = 3)
    plt.bar(br2, values_LV, width = bar_width, label = 'left ventricle', zorder = 3)

    plt.ylim((0.84,0.94))
    plt.title("Performance of different models", fontsize=20)
    plt.ylabel('DICE score', fontsize=15)
    plt.legend()
    plt.grid(axis = 'y', ls = '--', zorder = 0)
    plt.xticks([r + bar_width/2 for r in range(len(labels))],
        [l for l in labels], rotation = 0, fontsize = 15)

    plt.savefig("barchart.jpeg")

def plot_best_model_bar_chart():
    '''Plots bar chart of model performance on different resolutions '''

    data_avg = {'192_res': 0.8907, '384_res': 0.8939, '768_res': 0.8813}
    data_LV = {'192_res': 0.9054, '384_res': 0.9181, '768_res': 0.9169}
    labels = list(data_LV.keys())
    values_LV = list(data_LV.values())
    values_avg = list(data_avg.values())

    bar_width = 0.33
    br1 = np.arange(len(labels))
    br2 = [x + bar_width for x in br1]

    fig = plt.figure(figsize =(12, 8))
    plt.bar(br1, values_avg, width = bar_width, label = 'average', zorder = 3)
    plt.bar(br2, values_LV, width = bar_width, label = 'left ventricle', zorder = 3)

    plt.ylim((0.85,0.95))
    plt.title("Final model with different image resolutions", fontsize=20)
    plt.ylabel('DICE score', fontsize=15)
    plt.legend()
    plt.grid(axis = 'y', ls = '--', zorder = 0)
    plt.xticks([r + bar_width/2 for r in range(len(labels))],
        [l for l in labels], rotation = 0, fontsize = 20)

    plt.savefig("final_model_barchart.jpeg")


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

    plot_best_model_bar_chart()
    plot_bar_chart()
    plt.show()

        