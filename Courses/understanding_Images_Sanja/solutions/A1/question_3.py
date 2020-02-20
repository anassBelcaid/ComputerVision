"""
Script to get the figures for question 3
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import prewitt_h, prewitt_v
from skimage.io import imread, imsave
from scipy.ndimage import correlate
from skimage.transform import pyramid_gaussian
from skimage.transform  import rescale
from skimage.draw import rectangle_perimeter

def ESM(img):
    """
    Compute the edge strenght map of an imge
    """

    #Image in x
    img_x = prewitt_h(img)

    #imag ein y
    img_y = prewitt_v(img)

    return np.sqrt( img_x**2 + img_y**2)

def q_a():
    """
    Visualize the gradient magnitude for waldo and template
    """

    #reading the waldo image
    waldo = imread("./waldo.png", as_gray=True )


    #reading the template
    waldo_esm = ESM(waldo)
    imsave("waldo_ESM.png", waldo_esm)


    #template 
    template = imread("./template.png", as_gray=True)
    template_esm = ESM(template)
    imsave("template_ESM.png", template_esm)


def q_b():
    """
    Function to performe the corss corelation to find waldo
    """

    #IMG
    Img = imread("./waldo_ESM.png")
    Img = rescale(Img, 0.5)

    #template
    patch = imread("./template_ESM.png")
    patch = rescale(patch, 0.5)

    #pyramid for the template



    #scores
    scores = np.abs(correlate(Img, patch))
    # plt.imshow(scores, cmap = plt.cm.gray)


    imsave("./matching.png", scores)

    plt.show()


if __name__ == "__main__":
    
    #q_3
    # q_a()

    #q_b
    q_b()




