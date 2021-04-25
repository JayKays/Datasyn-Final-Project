
import os
import torch
import numpy as np

from matplotlib import pyplot as plt
from DatasetLoader import make_dataloaders, make_TEE_dataloader
from plotting import plot_segmentation
from utils import to_cuda, load_model
from evaluation import dice, class_dice
from Unet2D_default import default_Unet2D
from Unet2D_final import improved_Unet2D
from config import *


def test(model, dataset):
    '''Tests a given model with the diven dataloader,
    prints out average dice score and class-wise dice score,
    visualizes a few of the segmentation results'''

    if torch.cuda.is_available():
        model = model.cuda()

    model.train(False)
    
    #Loads dataset if a name is given in stead of dataloader
    if type(dataset) == str:
        if dataset == 'CAMUS_resized':
            _ , _ , test_data = make_train_dataloaders(dataset, (300,100,50))
    else:
        test_data = dataset
    
    print(f"Testing {MODEL_NAME} on {DATASET}")
    print('-'*10)

    #Result calculation over dataset
    acc = 0
    test_data.dataset.transform = False

    class_dices = np.zeros(4)
    for x, y in test_data:
        with torch.no_grad():
            predb = model(to_cuda(x))
        dice_score = dice(predb, to_cuda(y))
        class_score = class_dice(predb, to_cuda(y))

        class_score = class_score.detach().cpu().numpy()
        dice_score = dice_score.detach().cpu().numpy()

        acc += dice_score * x.shape[0]
        class_dices += class_score * x.shape[0]


    acc /= len(test_data.dataset)
    class_dices /= len(test_data.dataset)

    #Display results
    print(f'DICE score on test set: {np.round(acc, decimals = 4)}')
    print(f'Class-wise DICE score: {np.round(class_dices, decimals = 4)}')
    print('-'*60)

    # for i in range(len(test_data)):
    xb, yb = next(iter(test_data))

    with torch.no_grad():
            predb = model(to_cuda(xb))
    plot_segmentation(xb, yb, predb, num_img=4)
    plt.show()


if __name__ == "__main__":

    if DATASET != 'TTE':
        test_data = make_TEE_dataloader(DATASET)
    else:
        _, _, test_data = make_dataloaders(DATASET, DATA_SPLIT, transform = False)

    if MODEL_NAME == 'Baseline':
        unet = default_Unet2D(1, 4)
    else:
        unet = improved_Unet2D(1, 4)

    model_dict = load_model(MODEL_NAME, best = True)

    unet.load_state_dict(model_dict["model"])

    test(unet, test_data)