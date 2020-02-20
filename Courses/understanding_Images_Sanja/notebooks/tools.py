import numpy as np
import matplotlib.pyplot as plt

def Gaussian_kernel(sigma=1):
    """
    Gaussian kernel for the 1d case
    """

    #number of points after 4 stds
    N = int(3*sigma)

    #distances
    dx = np.r_[ np.arange(-N,0), np.arange(0,N+1)]

    #evaluating the expenontial
    cnst = 1/( np.sqrt(2*np.pi) * sigma)


    kernel = cnst*np.exp(-dx**2/(2*sigma**2))

    return kernel / np.sum(kernel)

def Gaussian_kernel_2d(sigma=1):
    """
    Gaussian kernel in the 2d case
    """

    #kernel 1
    #number of points after 4 stds
    N = int(3*sigma)

    #distances
    d = np.r_[ np.arange(-N,0), np.arange(0,N+1)]
    n = len(d)

    #diff in X
    X = np.tile(d, (n,1))
    Y = X.transpose()

    cst = 1/(2 * np.pi * sigma**2)

    Z = cst * np.exp(-(X**2 + Y**2)/(2*sigma**2))

    return X, Y, Z/ np.sum(Z)

def partial_Gaussian_X(sigma=1):

    """
    Partial Gaussin derivative
    """

    #kernel 1
    d = Gaussian_kernel(sigma)
    n = len(d)

    #diff in X
    X = np.tile(d, (n,1))
    Y = X.transpose()

    cst = 1/(2 * np.pi * sigma**4)

    Z = -cst * X * np.exp(-(X**2 + Y**2)/(2*sigma**2))

    return X, Y, Z/ np.sum(Z)
