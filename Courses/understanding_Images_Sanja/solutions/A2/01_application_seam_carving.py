"""
Script to visualize the results on Seam carving
"""

import numpy as np
import matplotlib.pyplot as plt
from seam_carving import energy_function, compute_cost, backtrack_seam, reduce
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_float


if __name__ == "__main__":
    
    #loading the image
    Img = imread("imgs/broadway_tower.jpg")
    Img = img_as_float(Img)


    #energy 
    energy = energy_function(Img)

    #costs
    v_costs, v_paths = compute_cost( Img, energy, axis=1)
    h_costs, h_paths = compute_cost( Img, energy, axis=0)


    #getting a seam
    end = np.argmin(v_costs[-1])
    seam = backtrack_seam(v_paths, end)

    #image with vseam
    vseam = np.copy(Img)
    for row in range(vseam.shape[0]):
            vseam[row, seam[row], :] = np.array([1.0, 0, 0])
    plt.imsave("vseam.png", vseam)


    #resizing
    H, W, _ = Img.shape
    W_new = 200

    out = reduce(Img, W_new)
    




    #plotting
    plt.imsave("broad_way_energy.png", energy, cmap = plt.cm.hot)
    plt.imsave("v_costs.png", v_costs, cmap = plt.cm.hot)
    plt.imsave("h_costs.png", h_costs, cmap = plt.cm.hot)
    plt.imsave("reduced.png", out)
    plt.show()




