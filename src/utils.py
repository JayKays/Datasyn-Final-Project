import numpy as np
import torch
from config import *
import os
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

def save_model(model, epoch, loss, optimizer = None, save_folder = None, model_name = None):
    model_dict = {}
    model_dict['model'] = model.state_dict()
    model_dict['epoch'] = epoch
    model_dict['loss'] = loss
    if optimizer != None:
        model_dict['optimizer'] = optimizer.state_dict()
    if save_folder is None:
        save_dir = os.path.join(SAVE_DIR, f'Last_Used_Checkpoints/epoch_{epoch}.pth')
    else:
        save_dir = os.path.join(SAVE_DIR, save_folder, model_name)
    # Save model
    torch.save(model_dict, save_dir)

def check_for_checkpoints():
    if not glob.glob(SAVE_DIR + '/*'):
        return False
    return True

def newest_checkpoint():
    files = glob.glob(SAVE_DIR + '/*')
    newest_model = max(files, key = os.path.getctime)
    return newest_model