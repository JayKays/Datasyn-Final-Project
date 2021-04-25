from pathlib import Path
from torch import nn
import torch
import albumentations as A
import os

#Hyper Parameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 12

#Loss function
LOSS_FUNC = nn.CrossEntropyLoss()

#Number of epochs without validation loss improvements before stopping
EARLY_STOP_TH = 10

#Name of dataset and split sizes of train/val/test
DATASET = 'TTE'
DATA_SPLIT = (4*350, 4*50, 4*50)

#Name of model to load/test/train
MODEL_NAME = 'Final_model'
# MODEL_NAME = 'Baseline'

#Paramters to decide wether to test, traing and/or Load model
TEST = True
TRAIN = False
LOAD = False

#Parameters to decide wether to extract TTE/TEE or not when running extraction script
EXTRACT_TTE = False
EXTRACT_TEE = False

#Path to directory with all datasets
BASE_PATH = Path("datasets")

#Path to original TTE and TEE directories
TTE_TRAIN_DIR = Path('datasets/CAMUS_full/training')
TEE_TRAIN_DIR = Path('datasets/DataTEEGroundTruth')

#Path to directories of extracted datasets 
# (Should contain train_gray and train_gt directories with images and ground truth in .tif format)
EXTRACTED_TTE_DIR = Path.joinpath(BASE_PATH, 'TTE')
EXTRACTED_TEE_DIR = Path.joinpath(BASE_PATH, 'TEE')

#Path to directory with saved models
MODEL_SAVE_DIR = Path("Saved_Models")

#Image processing
RESOLUTION = 384 # Default = 384, (must be a multiple of 32)
TRANSFORM = False

TRANSFORMS = A.Compose([
    # A.IAASharpen(alpha=(0.3, 0.7), lightness=(0.5, 1.0), always_apply=False, p=0.6),
    # A.RandomContrast (limit=(0, 0.5), always_apply=False, p=0.7),
    # A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(100,100), always_apply=False, p=0.5)
])

