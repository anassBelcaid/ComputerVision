import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    padH, padW = Hk//2, Wk//2
    image = zero_pad(image, padH, padW) 

    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            for i1 in range(Hk):
                for j1 in range(Wk):
                    out[i,j]+=kernel[i1, j1]*image[i+i1,j +j1]
    ### END YOUR CODE

    return out
def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))

    ### YOUR CODE HERE
    if(pad_width==0):
        out[pad_height:-pad_height,:]=image
    elif(pad_height==0):
        out[:,pad_width:-pad_width]=image
    else:
        out[pad_height:-pad_height, pad_width:-pad_width]=image
    ### END YOUR CODE
    return out
def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.
    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #flipping the kerne in x axis
    kernel = np.flip(kernel,axis=0);
    #flipping the kernel in the y axis
    # kernel = np.flip(kernel,axis=1)

    ### YOUR CODE HERE
    padded= zero_pad(image, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j]=np.sum(padded[i:i+Hk,j:j +Wk]* kernel)
    ### END YOUR CODE


    return out
def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out =conv_fast(f, np.flip(g, axis=0)) 
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g= g- g.mean()
    out = conv_fast(f, np.flip(g, axis=0))
    ### END YOUR CODE

    return out
def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    Hi, Wi = f.shape
    Hk, Wk =  g.shape
    out = np.zeros((Hi,Wi))
    ### YOUR CODE HERE
    padded= zero_pad(f, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            #taking the patch of the image
            patch_image = padded[i:i+Hk, j:j + Wk]

            #normalized patch of the template and the image
            m_p, s_p = g.mean(), g.std()
            m_i, s_i = patch_image.mean(), patch_image.std()
            patch_image = (patch_image - m_i)/ s_i 
            patch_kernel = (g - m_p) / s_p

            #cross correlation
            out[i,j]=np.sum(patch_image* patch_kernel)
    ### END YOUR CODE

    return out
