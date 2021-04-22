import numpy as np
import torch

from torch import nn

from DatasetLoader import make_dataloaders
from train import train
from evaluation import acc_metric, dice
from Unet2D_default import Unet2D
from Unet2D import imporoved_Unet2D
from test import test

from utils import *
from plotting import *
from config import *



def main ():

    bs = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    learn_rate = LEARNING_RATE
    base_path = BASE_PATH
    img_res = RESOLUTION

    should_load = LOAD
    should_train = TRAIN
    should_test = TEST

    model_name = 'Default_unet'

    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)

    #load the training data
    # train_data, valid_data = make_train_dataloaders('CAMUS', 8/9)
    train_data, valid_data, test_data = make_dataloaders('CAMUS', (4*300,4*100,4*50), img_res = img_res)

    #Model
    unet = Unet2D(1,4)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    
    #Loading from checkpoint
    start_epoch = 0
    if should_load:

        model_dict = load_model(model_name)
        unet.load_state_dict(model_dict["model"])

        start_epoch = model_dict["epoch"] + 1
        start_loss = model_dict["loss"]
    
    #Train model
    if should_train:
        train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, dice, model_name, start_epoch, epochs=num_epochs)

    if should_test:
        #Loads the best model under given name
        model_dict = load_model(model_name, best = True)
        unet.load_state_dict(model_dict["model"])

        #Tests on test set
        test(unet, test_data)


if __name__ == "__main__":
    main()