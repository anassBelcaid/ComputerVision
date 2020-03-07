"""
Here we are again, Image Stitching
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams['image.cmap'] = 'gray'

class MyStitcher():
    """
    Simple class to stitch two images together
    """

    def __init__(self, path1, path2):
        """
        Load and store the the imags in path1 and path2
        """
        self.img1 = cv2.imread(path1)
        self.img2 = cv2.imread(path2)

        #images in gray
        self.img1_g = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_g = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)


    def detect_and_compute(self):
        """
        Detect and compute SIFT features for both images
        """

        detector  = cv2.xfeatures2d.SIFT_create()
        self.kp1, self.f1 = detector.detectAndCompute(self.img1_g, None)
        self.kp2, self.f2 = detector.detectAndCompute(self.img2_g, None)


    def find_homography(self):
        """
        Find the homography transformation between image 1 and 2
        """

        #finding matches
        matcher = cv2.BFMatcher()
        matches = matcher.match(self.f2, self.f1)

        #sorting matches
        matches = sorted(matches, key = lambda x: x.distance)
        #converting to id
        matches =np.array( [[m.trainIdx, m.queryIdx] for m in matches],dtype='int')
        M1 = np.array([self.kp2[i].pt for i in matches[:,1]])
        M2 = np.array([self.kp1[i].pt for i in matches[:,0]])


        #computing the homography
        H, status = cv2.findHomography( M1, M2, method=cv2.RANSAC)

        #warp
        rows1, cols1 = self.img1_g.shape
        rows2, cols2 = self.img2_g.shape
        Img = cv2.warpPerspective(self.img2, H, dsize = (cols1 + cols2,rows1) )
        Img[:,:cols1] = self.img1
        plt.imshow(Img)
        plt.show()
        cv2.imwrite("panorama.png", Img)
        


if __name__ == '__main__':

    path1 = "./landscape_1.jpg"
    path2 = "./landscape_2.jpg"

    stitcher = MyStitcher(path1, path2)
    stitcher.detect_and_compute()

    stitcher.find_homography()
