import numpy as np
import torch
import glob

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from config import *

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True, img_res = 384):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.res= img_res
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_PIL = Image.open(self.files[idx]['gray']).resize((self.res, self.res))
        raw_us = np.stack([np.array(raw_PIL),], axis=2)

        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']).resize((self.res, self.res)))
        
        # raw_mask = np.where(raw_mask> 100, 1, 0)
        
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

def split_dataset(data, split):
    '''Splits up dataset, either by percentage (train/validation),
     or by a tuple with subset sizes (train/val/test)'''

    if type(split) == tuple:
        train_idx = range(split[0])
        valid_idx = range(train_idx[-1] + 1, split[0] + split[1])
        test_idx = range(valid_idx[-1] + 1, sum(split))

        train_data = Subset(data, train_idx)
        valid_data = Subset(data, valid_idx)
        test_data = Subset(data, test_idx)

        return train_data, valid_data, test_data
        
    else:
        train_idx = range(int(len(data)*split))
        valid_idx = range(train_idx[-1] + 1, len(data))

        train_data = Subset(data, train_idx)
        valid_data = Subset(data, valid_idx)

        return train_data, valid_data
    
    

def make_dataloaders(dataset, split, img_res = RESOLUTION):
    '''Makes dataset loaders for a diven dataset, either split into 
    train/validation or train/val/test depending on split parameter'''

    bs = BATCH_SIZE
    # bs = 12

    gt = Path.joinpath(BASE_PATH, dataset, 'train_gt')
    gray = Path.joinpath(BASE_PATH, dataset, 'train_gray')

    data = DatasetLoader(gray, gt, img_res = img_res)

    if type(split) == tuple and len(split) == 3 and sum(split) <= len(data):
        #Split dataset into training and validation
        train_data, valid_data, test_data = split_dataset(data, split)

        train_load = DataLoader(train_data, batch_size = bs, shuffle = True, num_workers = 4)
        valid_load = DataLoader(valid_data, batch_size = bs, shuffle = True, num_workers = 4)
        test_load = DataLoader(test_data, batch_size = bs, shuffle = False, num_workers = 4)

        return train_load, valid_load, test_load

    elif type(split) == float and 0 <= split <= 1:

        train_data, valid_data = split_dataset(data, split)

        train_load = DataLoader(train_data, batch_size = bs, shuffle = True, num_workers = 4)
        valid_load = DataLoader(valid_data, batch_size = bs, shuffle = True, num_workers = 4)

        return train_load, valid_load

    elif split == 'TEE':
        return DataLoader(data, batch_size = bs, shuffle = False, num_workers=4)

    else:
        raise (f"Training percentage = {split}, must be float between 0 and 1 or tuple of size 3 with proper subset sizes")

def make_TEE_dataloader(img_res = RESOLUTION):
    '''Loads the TEE dataset for testing'''
    
    TEE_data = make_dataloaders('TEE', 'TEE', img_res = img_res)

    return TEE_data