import numpy as np
import torch

from torch import nn

from DatasetLoader import make_train_dataloaders
from train import train
from evaluation import acc_metric, dice
from Unet2D_default import Unet2D
from Unet2D import imporoved_Unet2D
from test import test

from utils import *
from plotting import *
from config import *



def main ():
    visual_debug = VISUAL_DEBUG
    bs = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    learn_rate = LEARNING_RATE
    base_path = BASE_PATH

    should_load = LOAD
    should_train = TRAIN
    should_test = True

    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)

    #load the training data
    # train_data, valid_data = make_train_dataloaders('CAMUS', 8/9)
    train_data, valid_data, test_data = make_train_dataloaders('CAMUS', (4*300,4*100,4*50))

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    #Model
    unet = Unet2D(1,4)
    
    #Loading from checkpoint
    start_epoch = 0
    if should_load:

        model_dict = torch.load(model_path)
        unet.load_state_dict(newest_model["model"])

        start_epoch = newest_model["epoch"] + 1
        start_loss = newest_model["loss"]

        # print(f"...load complete. starting at epoch {start_epoch}")
    
    #Train model
    model_dir = 'test'
    if should_train:
        train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, dice, model_dir, start_epoch, epochs=num_epochs)

    if should_test:
        test(unet, test_data)

    # xb, yb = next(iter(valid_data))
    # with torch.no_grad():
    #     predb = unet(xb.cuda()[:15])

    # #show the predicted segmentations
    # if visual_debug:
    #     # plot_loss(train_loss, valid_loss)
    #     plot_segmentation(xb[:15], yb[:15], predb[:15])
    #     plt.show()


if __name__ == "__main__":
    main()