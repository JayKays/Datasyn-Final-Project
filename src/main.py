import numpy as np
import matplotlib as mp
import torch

from torch import nn

from DatasetLoader import make_dataloaders, make_TEE_dataloader
from train import train
from evaluation import acc_metric, dice
from Unet2D_default import default_Unet2D
from Unet2D import imporoved_Unet2D
from UNet import UNet
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

    model_name = MODEL_NAME
    dataset = DATASET
    split = DATA_SPLIT

    #sets the matplotlib display backend (most likely not needed)
    mp.use('TkAgg', force=True)

    #load the training data
    # train_data, valid_data = make_train_dataloaders('CAMUS', 8/9)
    if dataset == 'TEE':
        #Prevent accidentally training on TEE dataset, skips straight to test
        test_data = make_TEE_dataloader()
        should_train = False
        should_load = False

    #Model
    if model_name == 'Baseline':
        unet = default_Unet2D(1,4)
    else:
        unet = imporoved_Unet2D(1, 4)
    
    train_data, valid_data, test_data = make_dataloaders(dataset, split)

    #loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = LOSS_FUNC
    # opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    # opt = torch.optim.SGD(unet.parameters(), lr = learn_rate, momentum=0.99)
    opt = OPTIMIZER(unet.parameters(), lr=LEARNING_RATE)

    
    #Loading from checkpoint
    start_epoch = 0
    if should_load:

        model_dict = load_model(model_name)
        unet.load_state_dict(model_dict["model"])

        start_epoch = model_dict["epoch"]
        start_loss = model_dict["loss"]
    
    #Train model
    if should_train:
        train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, dice, model_name, start_epoch, epochs=num_epochs)

    if should_test:

        #Loads the best saved model under given name
        model_dict = load_model(model_name, best = True)
        unet.load_state_dict(model_dict["model"])

        #Tests model on test set
        test(unet, test_data)


if __name__ == "__main__":
    main()