import numpy as np
import os
import h5py
import glob

from medimage import image
from PIL import Image

from pathlib import Path

def TTE_image_gt(image_path, gt_path):
    '''Load an mhd image and its groudn truth
    from path and returns the PIL images'''

    #Loading as medimage.image
    img = image(image_path)
    gt = image(gt_path)
    
    #Convertin to PIL
    img = Image.fromarray(img.imdata.astype(np.uint8)[:,:,0], 'L')
    gt = Image.fromarray(gt.imdata.astype(np.uint8)[:,:,0], 'L')

    return img, gt

def TTE_image(image_path):
    '''Loads an mhd image from path and returns it as PIL'''

    #Loading as medimage.image
    img = image(image_path)

    #Convertin to PIL
    img = Image.fromarray(img.imdata.astype(np.uint8)[:,:,0], 'L')

    return img

def extract_TTE_to_tif(tte_dir, save_dir, data_type):
    '''Fetches all proper TTE mhd files and saves them in save directory as .tif format'''
    
    for image_type in ['_2CH_ED', '_2CH_ES', '_4CH_ED', '_4CH_ES']:
        for p in os.listdir(tte_dir):
            image_path = os.path.join(tte_dir, p, p + image_type + '.mhd')
            gt_path = os.path.join(tte_dir, p, p + image_type + '_gt.mhd')

            img_savepath = os.path.join(save_dir, data_type + '_gray', p + 'gray' + image_type + '.tif')
            gt_savepath = os.path.join(save_dir, data_type + '_gt', p + 'gt' + image_type + '.tif')

            if data_type == 'train':

                img, gt = TTE_image_gt(image_path, gt_path)

                img.save(img_savepath)
                gt.save(gt_savepath)

            elif data_type == 'test':
                img = TTE_image(image_path)

                img.save(img_savepath)


def extract_TEE_to_tif(TEE_path, save_dir):

    for d in os.listdir(TEE_path):
        files = glob.glob(str(Path.joinpath(TEE_path, d, '*.h5')))

        for h5 in files:
            print(h5)


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
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    
    extract_TTE = False
    extract_TEE = True

    if extract_TTE:
        tte_test_dir = Path("datasets/CAMUS_full/testing")
        tte_train_dir = Path("datasets/CAMUS_full/training")
        tte_save_dir = Path("datasets/TTE")

        #Makes dataset directories if they don't exist
        make_dir(Path.joinpath(tte_save_dir, 'train_gray'))
        make_dir(Path.joinpath(tte_save_dir, 'train_gt'))
        make_dir(Path.joinpath(tte_save_dir, 'test_gray'))

        extract_TTE_to_tif(tte_train_dir, tte_save_dir, 'train')
        extract_TTE_to_tif(tte_test_dir, tte_save_dir, 'test')

    if extract_TEE:
        # TEE_dir = Path('datasets/TEE_full')
        # TEE_save_dir = Path("datasets/TEE")

        # make_dir(Path.joinpath(TEE_save_dir, 'test_gray'))
        # make_dir(Path.joinpath(TEE_save_dir, 'test_gt'))
        
        # extract_TEE_to_tif(TEE_dir, TEE_save_dir)

        TEE_dir = Path('datasets/DataTEEGroundTruth')
        TEE_save_dir = Path('datasets/TEE_with_gt')

        make_dir(Path.joinpath(TEE_save_dir, 'train_gray'))
        make_dir(Path.joinpath(TEE_save_dir, 'train_gt'))

        extract_TEE_with_gt(TEE_dir, TEE_save_dir)
    






