
from pathlib import Path
import albumentations as A


NUM_EPOCHS = 1
<<<<<<< HEAD
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

=======
LEARNING_RATE = 0.01

BATCH_SZE = 12

BASE_PATH = Path('datasets/CAMUS_resized')
>>>>>>> 5014ea639fb1756f5e62e62a9646b6632062cb37
