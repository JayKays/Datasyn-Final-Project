import numpy as np
import torch
import glob

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image

from config import *

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

def make_data_loaders(data_splits, with_test = False):
    # torch.random.seed(1)

    bs = BATCH_SZE
    bs = 12

    gt = Path.joinpath(BASE_PATH, 'train_gt')
    gray = Path.joinpath(BASE_PATH, 'train_gray')

    data = DatasetLoader(gray, gt)

    if len(data_splits) == 2:
        #Split dataset into training and validation
        train_data, val_data = torch.utils.data.random_split(data, data_splits)

        train_load = DataLoader(train_data, batch_size = bs, shuffle = True)
        valid_load = DataLoader(val_data, batch_size = bs, shuffle = True)

        return train_load, valid_load

    elif len(data_splits) == 3:
        #Split dataset into train, validation and test
        train_data, val_data, test_data = torch.utils.data.random_split(data, data_splits)

        train_load = DataLoader(train_data, batch_size = bs, shuffle = True)
        valid_load = DataLoader(val_data, batch_size = bs, shuffle = True)
        test_load  = DataLoader(test_data, batch_size = bs, shuffle = True)

        return train_load, valid_load, test_load

    return DataLoader(data, batch_size = bs, shuffle = True)










