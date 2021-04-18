import numpy as np
import torch
import glob

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image

from config import *

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_files, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_files]
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


def split_data():
    pass


def make_data_loaders(data_splits, with_test = False):
    # BASE_PATH = 'datasets/CAMUS_resized'
    
    # files = glob.glob(BASE_PATH + '/train_gray/*.tif')
    files = [f for f in Path.joinpath(BASE_PATH, 'train_gray').iterdir() if not f.is_dir()]
    bs = BATCH_SZE
    bs = 12

    assert sum(data_splits) == len(files)

    train_size = data_splits[0]
    val_size = data_splits[1]
    test_size = data_splits[2]

    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]

    test_files = files[val_size + train_size:]

    gt = Path.joinpath(BASE_PATH, 'train_gt')

    train_data = DatasetLoader(train_files, gt)
    val_data = DatasetLoader(val_files, gt)
    test_data = DatasetLoader(test_files, gt)

    train_load = DataLoader(train_data, batch_size = bs, shuffle = True)
    val_load = DataLoader(val_data, batch_size = bs, shuffle = True)

    if data_splits[-1] != 0 and with_test:
        test_load = DataLoader(test_data, batch_size = bs, shuffle = True)
        return train_load, val_load, test_load

    return train_load, val_load










