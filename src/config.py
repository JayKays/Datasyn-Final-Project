from pathlib import Path
import albumentations as A
import os

NUM_EPOCHS = 10
LEARNING_RATE = 0.02
BATCH_SIZE = 12

VISUAL_DEBUG = True
LOAD = True
SAVE = True
<<<<<<< HEAD

BASE_PATH = Path("datasets")
SAVE_DIR = os.path.join(os.getcwd(), 'Saved_Models/')

RESOLUTION = 384
=======
BASE_PATH = Path("./datasets/CAMUS_resized")
SAVE_DIR = os.path.join(os.getcwd(), 'saved_models/')
>>>>>>> 169a0c4b35ce3af6c372fb57345a33a899a6e4b3

PREPROCESS_RECIPE = ['gaussian',
                    #'bilateral'
                    #'rotate',
                    #'resize'
                    ]


PREPROCESS_PARAMS = {
    # for gaussian blur; 2 by default
    "radius": 2,
    "rotate": 90,
    "resize": (384, 384),
    
    # for bilateral smoothing
    "kernel_size": 15,
    "sig_color": 15,
    "sig_space": 15,
}

