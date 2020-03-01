"""
Feature Extraction and homography with the SIFT method
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from panorama import match_descriptors
plt.rcParams['image.cmap'] = 'gray'

def get_sift_features(Img):
    """
    Wrapper function to get the SIFT features on the image
    """

    if Img.ndim == 3:
        #convert the image to grayscale
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    #¢reating the sift instance
    sift = cv2.xfeatures2d.SIFT_create()

    #getting the keypoints
    kp, desc = sift.detectAndCompute(Img, None)

    return kp, desc

def q1():
    """
    Question 1 compute the SIFT keypoints
    """

    #filenames
    names = ["reference.png", "test.png", "test2.png"]
    sift_names  = ['ref_sift.png', 'test_sift.png', 'test2_sift.png']

    #images
    Imgs = list(map(cv2.imread, names))

    for img, siftname in zip(Imgs, sift_names):
        print("Processing img : {}".format(siftname))
        kp, _ = get_sift_features(img)
        out = img.copy()
        cv2.drawKeypoints(img, kp[-100:], outImage = out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imsave(siftname, out)

def q2():
    """
    Function to draw the matches
    """
    template = cv2.imread("./reference.png")
    kp1,desc1 = get_sift_features(template) 

    #reference
    ref = cv2.imread("./test.png")
    kp2, desc2 = get_sift_features(ref)


    #reference 2
    ref2 = cv2.imread("./test2.png")
    kp3, desc3 = get_sift_features(ref2)

    #computing the matches
    matchs = match_descriptors(desc1, desc2)
    Dmatches = []
    #convert matchs to D Matchs
    for match  in matchs:
        i, j = match
        D = np.sqrt(np.sum((desc1[i]- desc2[j])**2))
        Dmatches.append( cv2.DMatch(match[0], match[1], D ))



    #plotting the matches
    Out = cv2.drawMatches(template, kp1, ref, kp2, Dmatches[:15], None)
    cv2.imwrite("simple_matching1.png", Out)
    plt.imshow(Out)
    plt.show()

    matchs = match_descriptors(desc1, desc3)
    Dmatches = []
    #convert matchs to D Matchs
    for match  in matchs:
        i, j = match
        D = np.sqrt(np.sum((desc1[i]- desc3[j])**2))
        Dmatches.append( cv2.DMatch(match[0], match[1], D ))



    #plotting the matches
    Out = cv2.drawMatches(template, kp1, ref2, kp3, Dmatches[:15], None)
    cv2.imwrite("simple_matching2.png", Out)
    plt.imshow(Out)
    plt.show()
    

def find_affine_transform(kp1, kp2, desc1, desc2, matchs):
    """
    Function to find the affine transform between the set of kp1 and kp2
    """


    X = np.float32([kp1[i].pt for i in matchs[:3,0]])
    Y = np.float32([kp2[i].pt for i in matchs[:3,1]])


    #compute the transofmration matrix
    M = cv2.getAffineTransform(X, Y)

    return M

def q3():
    """
    Find affine transformation between template and figure
    """

    template = cv2.imread("./reference.png")
    kp1,desc1 = get_sift_features(template) 

    #reference
    ref = cv2.imread("./test2.png")
    row, col, ch = ref.shape
    kp2, desc2 = get_sift_features(ref)



    #matching desciptors
    matchs = match_descriptors(desc1, desc2)

    #getting the keypoints as (x, y)
    H = find_affine_transform(kp1, kp2, desc1, desc2, matchs)

    #Drawing affine transform
    dst = cv2.warpAffine(template, H, (col, row))
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(dst)
    axs[0].axis('off')
    axs[1].imshow(ref)
    axs[1].axis('off')
    plt.savefig("linear_transform_test2.pdf",bbox_inches="tight")
    plt.show()







if __name__ == '__main__':
    #question 1 Compute sift for reference, test, and test2.png
    # q1()


    #question 2: match elements
    # q2()

    #question 3: Affine transforms
    q3()
