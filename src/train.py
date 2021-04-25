import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from evaluation import acc_metric, class_dice
from utils import *
from config import *

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, model_dir, start_epoch=0, epochs=1, model_dict = None):
    start = time.time()
    model.cuda()

    if not LOAD:
        model_dir = make_model_dir(Path.joinpath(MODEL_SAVE_DIR, model_dir))
    else:
        model_dir = Path.joinpath(MODEL_SAVE_DIR, model_dir)

    if model_dict is not None:
        train_loss = to_cuda(model_dict['train_loss'])
        valid_loss = to_cuda(model_dict['valid_loss'])
    else:
        train_loss, valid_loss = [], []

    best_model_path = Path.joinpath(model_dir)
    model_save_path = Path.joinpath(model_dir)


    best_LV_acc = 0.0
    min_loss = np.inf
    early_stop_counter = 0

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        epoch_start = time.time()

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0
            running_class_dice = to_cuda(torch.zeros(4))

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()

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
                

                running_acc  += acc * x.shape[0]
                running_loss += loss * x.shape[0]

                running_class_dice += class_dice(outputs, y) * x.shape[0]
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_class_dice = running_class_dice / len(dataloader.dataset)
            epoch_class_dice = epoch_class_dice.detach().cpu().numpy()

            epoch_time = time.time() - epoch_start

            print('{} Loss: {:.4f} DICE: {:.4f} Class DICE: {}'.format(phase, epoch_loss, epoch_acc, np.round(epoch_class_dice, decimals = 4)))
            # print(f'Class-wise DICE: {epoch_class_dice[:]}')
            if phase == "valid":
                #Updates best performing model for LV class
                if best_LV_acc <= epoch_class_dice[1]:
                    save_model(model, best_model_path, 'best_model', epoch, to_cpu(train_loss), to_cpu(valid_loss))
                    best_LV_acc = epoch_class_dice[1]

                #Increments or resets early stopping counter
                if min_loss <= epoch_loss:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                    min_loss = epoch_loss
            
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

        #ETA calculation
        avg_epoch_time = (time.time() - start) / (epoch - start_epoch + 1)
        eta = (epochs - (epoch + 1))*avg_epoch_time

        print('Epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60)) 
        print('ETA: {:.0f}m {:.0f}s'.format(eta // 60, eta % 60))
        print('-' * 60)

        #Saves checkpoint avery 5 epochs
        if epoch % 5 == 0:
            print(f"Saved Checkpoint")
            save_model(model, model_save_path, 'last_checkpoint',epoch + 1, to_cpu(train_loss), to_cpu(valid_loss))
            print('-'*60)

        if early_stop_counter > EARLY_STOP_TH: 
            print("Early stopped!")
            print('-'*60)
            break
    
    #Saves last model after training
    save_model(model, model_save_path, 'last_checkpoint', epoch + 1, to_cpu(train_loss), to_cpu(valid_loss))
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    print('-'*60)

    return to_cpu(train_loss), to_cpu(valid_loss)
