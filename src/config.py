from pathlib import Path
# from dice_loss import dice_loss
from torch import nn
import torch
import albumentations as A
import os


#Hyper Parameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
MOMENTUM = 0
BATCH_SIZE = 12

LOSS_FUNC = nn.CrossEntropyLoss()

#Number of epochs without validation loss improvements before stopping
EARLY_STOP_TH = 4

#Name of dataset and split sizes of train/val/test
DATASET = 'TTE'
DATA_SPLIT = (4*300, 4*100, 4*50)

#Name of model to load/test/train
MODEL_NAME = 'Baseline'

#Paramters to decide wether to test, traing and/or Load model
TEST = True
TRAIN = True
LOAD = False

#Parameters to decide wether to extract TTE/TEE or not when running exstraxtion script
EXTRACT_TTE = False
EXTRACT_TEE = False

#Dataset paths
BASE_PATH = Path("datasets")

#Path to original TTE and TEE directories
TTE_TRAIN_DIR = Path('datasets/CAMUS_full/training')
TEE_TRAIN_DIR = Path('datasets/DataTEEGroundTruth')

#Path to directories to store extracted datasets
EXTRACTED_TTE_DIR = Path.joinpath(BASE_PATH, 'TTE')
EXTRACTED_TEE_DIR = Path.joinpath(BASE_PATH, 'TEE')

#Path to directory with saved models
MODEL_SAVE_DIR = Path("Saved_Models")

#Image processing
RESOLUTION = 384
TRANSFORM = True

TRANSFORMS = A.Compose([
    A.IAASharpen(alpha=(0.1, 0.4), lightness=(0.5, 1.0), always_apply=False, p=0.5),
])

# PREPROCESS_RECIPE = ['gaussian',
#                     #'bilateral'
#                     #'rotate',
#                     #'resize'
#                     ]

# PREPROCESS_PARAMS = {
#     # for gaussian blur; 2 by default
#     "radius": 2,
#     "rotate": 90,
#     "resize": (384, 384),
    
#     # for bilateral smoothing
#     "kernel_size": 15,
#     "sig_color": 15,
#     "sig_space": 15,
# }
