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
from evaluation import acc_metric, dice
from plotting import *
from utils import *
from config import *

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, start_epoch=0, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        epoch_start = time.time()

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0
            running_dice = 0.0

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
                tot_dice, batch_score = dice(outputs, y) 

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                running_dice += tot_dice

                # print(tot_dice)

                # print(step)
                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader)

            epoch_time = time.time() - epoch_start

            #ETA calculation
            avg_epoch_time = (time.time() - start) / (epoch - start_epoch + 1)
            eta = (epochs - (epoch + 1))*avg_epoch_time


            # print('Epoch {}/{}'.format(epoch+1, epochs))
            # print('-' * 10)
            print('{} Loss: {:.4f} Acc: {:.4f} DICE: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_dice))
            print('-' * 10)
            
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

        #Saves checkpoint avery 10 epochs
        if SAVE and (epoch + 1) % 10 == 0:
            print(f"saving model with epoch = {epoch}, loss = {loss}...")
            save_model(model, epoch, epoch_loss)
            print("...save complete.")
            

        print('Epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60), end='\t') 
        print('ETA: {:.0f}m {:.0f}s'.format(eta // 60, eta % 60))
        print('-' * 60)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    print(f"Saving Model", end = '...')
    save_model(model, epoch, epoch_loss, save_folder="Default_UNET", model_name="CAMUS_resized.pth")
    print("...save complete.")

    cpu_train_loss = [x.detach().cpu().item() for x in train_loss]
    cpu_valid_loss = [x.detach().cpu().item() for x in valid_loss]

    return cpu_train_loss, cpu_valid_loss
