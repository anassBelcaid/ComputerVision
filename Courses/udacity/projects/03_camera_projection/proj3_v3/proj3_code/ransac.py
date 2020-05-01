import numpy as np
import math
from proj3_code.least_squares_fundamental_matrix import solve_F
from proj3_code import two_view_data
from proj3_code import fundamental_matrix
from fundamental_matrix import point_line_distance


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    ##############################
    # TODO: Student code goes here
    return np.log(1- prob_success)/ np.log(1 - ind_prob_correct**sample_size)
    ##############################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. CoHowever, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """
    ##############################
    # TODO: Student code goes here
    n = x_0s.shape[0]
    if x_0s.shape[1] == 2:
        x_0s_3d = np.c_[x_0s, np.ones(n)]
        x_1s_3d = np.c_[x_1s, np.ones(n)]
    else:

        x_0s_3d = x_0s
        x_1s_3d = x_1s

    lines = (x_1s_3d @ F.T)

    inliers = np.zeros(n)
    for i in range(n):
        inliers[i] = point_line_distance(lines[i], x_0s_3d[i])

    inliers = np.nonzero(inliers < threshold)[0]

    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """
    ##############################
    # TODO: Student code goes here
    #probability of sucess
    P = 0.9

    #sample size
    n = len(x_0s)
    k = max(int(n/50), 6)

    #probability of selection
    p=0.1

    #number of repetition
    S = int(calculate_num_ransac_iterations(P,k,p))
    print(S)

    #initial number of inliers
    num_inliers = 0

    indices = np.arange(n)
    for iter in range(1000):
        #sampling
        sample = np.random.choice(indices, k, p)

        #left and right points
        left_p, right_p = x_0s[sample], x_1s[sample]

        #fitting
        F = solve_F(left_p, right_p)

        #find inliers
        inliers = find_inliers(x_0s, F, x_1s, 1)

        if len(inliers) > num_inliers:
            best_F = F
            num_inliers = len(inliers)


    inliers_x_0, inliers_x_1 = x_0s[inliers], x_1s[inliers]

    ##############################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """
    ##############################
    # TODO: Student code goes here
    raise NotImplementedError
    ##############################
