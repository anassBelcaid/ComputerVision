"""
Implement and visualize the harris corner
"""

from panorama import harris_corners
from skimage.io import imread
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt




if __name__ == "__main__":
    
    #reading the image
    Img = imread("./building.jpg", as_gray=True)

    #computing the response
    response = harris_corners(Img)


    corners = corner_peaks(response, threshold_rel=0.05)

    # Display detected corners
    plt.imshow(Img,cmap=plt.cm.gray)
    plt.scatter(corners[:,1], corners[:,0], marker='x')
    plt.axis('off')
    plt.title('Detected Corners')
    plt.savefig("./harris_response.png",cmap=plt.cm.gray)
    plt.show()

