import numpy as np
import torch
from config import *
import os
from pathlib import Path
import glob

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

def make_model_dir(dir_path):
    model_number = 1
    temp_path = dir_path

    while os.path.exists(temp_path):
        temp_path = Path(str(dir_path) + f'{model_number}')
        model_number += 1

    try:
        os.makedirs(temp_path)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    
    return temp_path
def save_model(model, save_dir, model_name, epoch, loss, optimizer = None):
    model_dict = {}
    model_dict['model'] = model.state_dict()
    model_dict['epoch'] = epoch
    model_dict['loss'] = loss

    if optimizer != None:
        model_dict['optimizer'] = optimizer.state_dict()
    
    save_path = Path.joinpath(save_dir, model_name + '.pth')

    # Save model
    torch.save(model_dict, save_path)

def check_for_checkpoints():
    if not glob.glob(SAVE_DIR + '/*'):
        return False
    return True

def newest_checkpoint():
    files = glob.glob(SAVE_DIR + '/*')
    newest_model = max(files, key = os.path.getctime)
    return newest_model