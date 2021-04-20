import numpy as np
import os

from medimage import image
from PIL import Image

from pathlib import Path


def TTE_image_gt(image_path, gt_path):
    'Load an mhd image and returns it as PIL'

    #Loading as medimage.image
    img = image(image_path)
    gt = image(gt_path)
    

    #Convertin to PIL
    img = Image.fromarray(img.imdata.astype(np.uint8)[:,:,0], 'L')
    gt = Image.fromarray(gt.imdata.astype(np.uint8)[:,:,0], 'L')

    return img, gt


def save_TTE_data_to_tif(tte_dir, save_dir):
    
    for image_type in ['_2CH_ED', '_2CH_ES', '_4CH_ED', '_4CH_ES']:
        for p in os.listdir(tte_dir):
            image_path = os.path.join(tte_dir, p, p + image_type + '.mhd')
            gt_path = os.path.join(tte_dir, p, p + image_type + '_gt.mhd')
            
            img, gt = TTE_image_gt(image_path, gt_path)

            img_savepath = os.path.join(save_dir, 'train_gray', p + 'gray' + image_type + '.tif')
            gt_savepath = os.path.join(save_dir, 'train_gt', p + 'gt' + image_type + '.tif')

            img.save(img_savepath)
            gt.save(gt_savepath)

if __name__ == "__main__":
    
    tte_dir = Path("datasets/CAMUS_full/training")
    save_dir = Path("datasets/CAMUS")

    save_TTE_data_to_tif(tte_dir, save_dir)




