import numpy as np
import torch

from torch import nn

from DatasetLoader import make_train_dataloaders
from train import train
from evaluation import acc_metric
from Unet2D import Unet2D
from test import test

from utils import *
from plotting import *
from config import *



def main ():
    visual_debug = VISUAL_DEBUG
    bs = BATCH_SIZE
    epochs_val = NUM_EPOCHS
    learn_rate = LEARNING_RATE
    base_path = BASE_PATH

    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)

    #load the training data
    # train_data, valid_data = make_train_dataloaders('CAMUS', 8/9)
    train_data, valid_data, test_data = make_train_dataloaders('CAMUS_resized', (300,100,50))
    # print(len(test_data), len(test_data.dataset))
    

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,2)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    
    #Loading from checkpoint
    start_epoch = 0
    if LOAD and check_for_checkpoints():

        newest_file = newest_checkpoint()
        newest_model = torch.load(newest_file)

        unet.load_state_dict(newest_model["model"])
        # opt.load_state_dict(newest_model["optimizer"])

        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        start_epoch = newest_model["epoch"] + 1
        start_loss = newest_model["loss"]
        # start_epoch += 1
        print(f"...load complete. starting at epoch {start_epoch}")
    # else:
        # print("LOAD == False or no checkpoints found.")
    # print(f"start Loss = {start_loss:.4f}")


    #predict on the next train batch (is this fair?)
    # xb, yb = next(iter(train_data))
    # with torch.no_grad():
    #     predb = unet(xb)
    
    # if visual_debug:
    #     # plot_loss(train_loss, valid_loss)
    #     plot_segmentation(bs, xb, yb, predb)
    #     plt.show()
    
    #do some training
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, start_epoch, epochs=epochs_val)

    #plot training and validation losses

    #predict on the next train batch (is this fair?)
    test(unet, 'CAMUS_resized')

    # xb, yb = next(iter(test_data))
    # with torch.no_grad():
    #     predb = unet(xb.cuda()[:15])

    # #show the predicted segmentations
    # if visual_debug:
    #     # plot_loss(train_loss, valid_loss)
    #     plot_segmentation(xb[:15], yb[:15], predb[:15])
    #     plt.show()

if __name__ == "__main__":
    main()