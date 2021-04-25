import numpy as np
import torch
import os
import glob

from pathlib import Path
from config import *

def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements

def to_cpu(elements):
    """
    Transfers every object in elements to CPU if cuda is available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.detach().cpu().item() for x in elements]
        return elements.detach().cpu().item()
    return elements


def make_model_dir(dir_path):
    '''Makes a new model directory for a given path, 
    to prevent overwriting existing models with the same name'''

    model_number = 1
    temp_path = dir_path

    while os.path.exists(temp_path) and len(os.listdir(temp_path)) != 0:
        temp_path = Path(str(dir_path) + f'{model_number}')
        model_number += 1

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    return temp_path

def save_model(model, save_dir, model_name, epoch, train_loss, valid_loss, optimizer = None):
    '''Saves important paramters for a given model, 
    to be able to load and pickup training and test the model'''

    model_dict = {}
    model_dict['model'] = model.state_dict()
    model_dict['epoch'] = epoch
    model_dict['train_loss'] = train_loss
    model_dict['valid_loss'] = valid_loss

    if optimizer != None:
        model_dict['optimizer'] = optimizer.state_dict()

    if not os.path.exists(save_dir):
        save_dir = make_model_dir(save_dir)
    
    save_path = Path.joinpath(save_dir, model_name + '.pth')

    # Save model
    torch.save(model_dict, save_path)


def load_model(model_name, best = False):
    '''Loads the model dictonary saved with the given model name'''

    if best:
        load_path = Path.joinpath(MODEL_SAVE_DIR, model_name,'best_model.pth')
    else:
        load_path = Path.joinpath(MODEL_SAVE_DIR, model_name, 'last_checkpoint.pth')

    #Checks for existence of model directory
    assert os.path.exists(load_path)

    return torch.load(load_path)
