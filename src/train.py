import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from DatasetLoader import DatasetLoader, make_data_loaders
from Unet2D import Unet2D
from evaluation import acc_metric, dice_score
from plotting import *
from utils import *

from config import *

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                print(step)
                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss.detach().cpu().item()) if phase=='train' else valid_loss.append(epoch_loss.detach().cpu().item())

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def main ():
    torch.manual_seed(0)

    
    #enable if you want to see some plotting
    visual_debug = True

    #batch size
    bs = BATCH_SZE

    #epochs

    epochs_val = NUM_EPOCHS


    #learning rate
    learn_rate = LEARNING_RATE

    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)

    #load the training data
    base_path = BASE_PATH

    train_data, valid_data = make_data_loaders((300,150))

    xb, yb = next(iter(train_data))
    print (xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,2)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    #do some training
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, epochs=epochs_val)

    #predict on the next train batch (is this fair?)
    xb, yb = next(iter(train_data))
    with torch.no_grad():
        predb = unet(xb.cuda())

    #show the predicted segmentations and loss
    if visual_debug:
        plot_loss(train_loss, valid_loss)
        plot_visual_results(bs, xb, yb, predb)
        plt.show()

if __name__ == "__main__":
    main()
