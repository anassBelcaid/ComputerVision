"""
Script to produce the figures for question 2
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.filters import gaussian


def q_c():
    """
    convolving with waldo and sigma 1
    """

    #reading the image
    waldo = imread("./waldo.png", as_gray=False)


    #convolving
    wald_conv = gaussian(waldo, sigma = 1)

    #saving the image
    imsave("waldo_gauss_c.png", wald_conv)



if __name__ == "__main__":
    
    #question C
    q_c()
