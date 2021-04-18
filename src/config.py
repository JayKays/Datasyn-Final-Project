
from pathlib import Path
import albumentations as A


NUM_EPOCHS = 1
LEARNING_RATE = 0.04
BATCH_SIZE = 12

VISUAL_DEBUG = True

PREPROCESS_RECIPE = ['gaussian',
                    #'BilateralSmooth'
                    #'MedianFilter',
                    ]

PREPROCESS_PARAMS = {
    # for gaussian blur; 2 by default
    "radius": 2,
    
    # for bilateral smoothing
    "kernel_size": 15,
    "sig_color": 15,
    "sig_space": 15,
}

