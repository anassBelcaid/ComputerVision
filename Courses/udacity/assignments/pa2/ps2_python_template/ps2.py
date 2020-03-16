# ps2
import os
import numpy as np
from skimage.io import imread, imsave
import skimage
import matplotlib.pyplot as plt



## 1-a
# Read images
def q1():
    """
    Simple disparity map with SSD
    """
    L = skimage.img_as_float(imread("./input/pair0-L.png", as_gray=True))
    R = skimage.img_as_float(imread("./input/pair0-R.png", as_gray=True))

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    from disparity_ssd import disparity_ssd
    D_L = disparity_ssd(L, R)
    imsave("./output/ps2-1-a-1.png", D_L)
    D_R = disparity_ssd(R, L)
    imsave("./output/ps2-1-a-2.png", D_R)

if __name__ == "__main__":
    #First Question
    q1()

