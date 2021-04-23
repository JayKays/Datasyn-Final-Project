import numpy as np
import os
import h5py
import glob

from medimage import image
from PIL import Image

from pathlib import Path
from matplotlib import pyplot as plt
from config import *


def mhd_to_PIL(img_path, new_spacing = [0.154, 0.154]):
    '''Loads an mhd image from path and returns it as PIL'''

    #Loading as medimage.image
    img = image(img_path)

    spacing = img.spacing()
    dims = img.header['DimSize']
    # print(spacing)

    new_x = np.round(spacing[0] * dims[0] / new_spacing[0]).astype(int)
    new_y = np.round(spacing[1] * dims[1] / new_spacing[1]).astype(int)

    #Convertin to PIL
    PIL_img = Image.fromarray(img.imdata.astype(np.uint8)[:,:,0], 'L').resize((new_x, new_y))

    # fig, ax = plt.subplots(1, 2, figsize=(5,10))
    # ax[1].imshow(np.asarray(PIL_img))
    # ax[0].imshow(img.imdata.astype(np.uint8)[:,:,0])
    # plt.show()

    return PIL_img

def extract_TTE_to_tif(tte_dir, save_dir):
    '''Fetches all proper TTE mhd files and saves them in save directory as .tif format'''
    
    for p in os.listdir(tte_dir):
        for image_type in ['_2CH_ED', '_2CH_ES', '_4CH_ED', '_4CH_ES']:

            #Path to original dataset
            image_path = os.path.join(tte_dir, p, p + image_type + '.mhd')
            gt_path = os.path.join(tte_dir, p, p + image_type + '_gt.mhd')

            #Path to save extracted images
            img_savepath = os.path.join(save_dir, 'train_gray', p + 'gray' + image_type + '.tif')
            gt_savepath = os.path.join(save_dir, 'train_gt', p + 'gt' + image_type + '.tif')

            #Convert mhd images to PIL
            img = mhd_to_PIL(image_path)
            gt = mhd_to_PIL(gt_path)
            # return
            #Save
            img.save(img_savepath)
            gt.save(gt_savepath)

        #Stops after 450 patients, as the last 50 are empty
        if p == 'patient0450': break


def extract_TEE_with_gt(TEE_dir, save_dir):
    img_path = Path.joinpath(TEE_dir, 'train_gray')
    gt_path = Path.joinpath(TEE_dir, 'train_gt')

    img_save_dir = Path.joinpath(save_dir, 'train_gray')
    gt_save_dir = Path.joinpath(save_dir, 'train_gt')

    image_files = glob.glob(str(img_path) + '/*')
    gt_files = glob.glob(str(gt_path) + '/*')

    for i in range(len(image_files)):

        #Making image save path(removes .jpg adds .tif)
        img_name = image_files[i].split('\\')[-1]
        img_name = img_name[:-4]
        img_save_path = Path.joinpath(img_save_dir, img_name + '.tif')
        
        #Making gt save_path (Removes first gt_ from name)
        gt_name = gt_files[i].split('\\')[-1]
        gt_name = gt_name[3:]
        gt_save_path = os.path.join(gt_save_dir, gt_name)

        #Open img
        img = Image.open(image_files[i])
        
        #Open and convert gt to 0,1,2
        gt = np.array(Image.open(gt_files[i]).convert('L'))
        gt = np.round(gt/127.5)
        gt = Image.fromarray(gt.astype(np.uint8), 'L')

        #Rotate to match TTE
        img = img.rotate(180)
        gt = gt.rotate(180)

        #Save
        img.save(img_save_path)
        gt.save(gt_save_path)


def make_dir(dir_path):
    '''Create a diven directory if it doesn't exist'''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    
    extract_TTE = EXTRACT_TTE
    extract_TEE = EXTRACT_TEE

    if extract_TTE:
        tte_train_dir = TTE_TRAIN_DIR
        tte_save_dir = EXTRACTED_TTE_DIR

        #Makes dataset directories if they don't exist
        make_dir(Path.joinpath(tte_save_dir, 'train_gray'))
        make_dir(Path.joinpath(tte_save_dir, 'train_gt'))

        extract_TTE_to_tif(tte_train_dir, tte_save_dir)

    if extract_TEE:
        TEE_dir = TEE_TRAIN_DIR
        TEE_save_dir = EXTRACTED_TEE_DIR

        #Makes directories if they don't exist
        make_dir(Path.joinpath(TEE_save_dir, 'train_gray'))
        make_dir(Path.joinpath(TEE_save_dir, 'train_gt'))

        extract_TEE_with_gt(TEE_dir, TEE_save_dir)
    






