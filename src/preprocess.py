import numpy as np
import pandas as pd

from PIL import Image, ImageFilter, ImageEnhance
import cv2

from utils import *
from plotting import *
from config import *

def preprocess(image, preprocess_recipe, params):
    pre_im = image.copy()
    kernel_size = params["kernel_size"]
    sig_color = params["sig_color"]
    sig_space = params["sig_space"]
    radius = params["radius"]
    
    for prep in preprocess_recipe:
        if prep == "gaussian":
            print("applying gaussian")
            pre_im = pre_im.filter(ImageFilter.GaussianBlur(radius))
        if prep == "bilateral":
            print("applying bilateral")
            bilateral=cv2.bilateralFilter(np.array(image_pre), kernel_size, sig_color, sig_space)
            pre_im = Image.fromarray(np.uint8(bilateral)).convert('L')
    
    return pre_im

if __name__ == "__main__": # for testing
    image = Image.open("test.jpeg")
    #image.show()
    image.save("original.jpeg", "jpeg")
    image = preprocess(image, PREPROCESS_RECIPE, PREPROCESS_PARAMS)
    image.save("preprocessed.jpeg", "jpeg")
    #image.show()
    print("done")