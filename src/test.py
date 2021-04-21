
import os
import torch
import numpy as np

from matplotlib import pyplot as plt
from DatasetLoader import make_test_dataloader, make_train_dataloaders
from plotting import plot_segmentation
from utils import to_cuda
from evaluation import dice
from Unet2D_default import Unet2D
from Unet2D import imporoved_Unet2D
from config import *


def test(model, dataset):

    if type(dataset) == str:
        if dataset == 'CAMUS_resized':
            _ , _ , test_data = make_train_dataloaders(dataset, (300,100,50))
        else:
            test_data = make_test_dataloader(dataset)
    else:
        test_data = dataset
    
    acc = 0
    class_dices = np.zeros(4)

    for x, y in test_data:

        with torch.no_grad():
            predb = model(x.cuda())
        
        tot_dice, class_dice = dice(predb, to_cuda(y))

        class_dice = class_dice.detach().cpu().numpy()
        tot_dice = tot_dice.detach().cpu().numpy()
        
        acc += tot_dice * test_data.batch_size
        class_dices += class_dice * test_data.batch_size

    acc /= len(test_data.dataset)
    class_dices /= len(test_data.dataset)

    print(f'DICE score on test set: {np.round(acc, decimals = 4)}')
    print(f'Class wise DICE score: {np.round(class_dices[1:], decimals = 4)}')

    xb, yb = next(iter(test_data))
    with torch.no_grad():
            predb = model(xb.cuda())

    plot_segmentation(xb, yb, predb)
    plt.show()


if __name__ == "__main__":

    _, _, test_data = make_train_dataloaders('CAMUS', (4*300,4*100,4*50))

    unet = Unet2D(1,4).cuda()

    test(unet, test_data)