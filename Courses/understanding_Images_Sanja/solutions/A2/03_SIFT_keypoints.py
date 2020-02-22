"""
Function to detect the SIFT keypoint and plot them
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sift import SIFT
import cv2 as cv


if __name__ == "__main__":
    
    #reading the image

    Img = imread("./building.jpg")


    #creating SIFT features could change number of octave and other
    # parameters
    detector = SIFT(Img, n_oct=8)

    #detecting keypoint
    detector.detect()

    keypoints = detector.get_keypoints()


    #drawing the keypoints
    OutImage = np.zeros_like(Img)
    cv.drawKeypoints(Img, keypoints, OutImage)
    plt.imshow(OutImage)
    plt.show()

