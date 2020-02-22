import numpy as np
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from collections import defaultdict
from cv2 import circle
import cv2
from skimage.color import rgb2gray, gray2rgb
from numpy import sin, cos
def gaussian_kernel_1d(sigma=1):
    """
    retun a gaussian kernel of size 2*L+1 with std sigma
    """
    
    #integer points
    limit = np.ceil(4*sigma)
    points = np.unique(np.floor(np.linspace(-limit, limit)))

    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-points**2/sigma**2)
def gaussian_kernel(sigma=1):
    """
    Gaussian for 2d image with more than on channel
    """
    K = gaussian_kernel_1d(sigma)

    K = np.outer(K,K)
    K = K[:,:,np.newaxis]

    return K
def upsample_1d(sig, factor=2):
    """
    interpoloate the signal sig with factor 
    """

    size_sig = len(sig)
    # resulting signal
    sig_upsampled = np.zeros(size_sig * factor)

    #copying the initial points
    sig_upsampled[::factor] = sig

    #convolving with a kernel
    kernel = np.ones(factor)
    kernel = (1/factor)*np.convolve(kernel, kernel)

    #return the computed array
    return np.convolve(sig_upsampled, kernel, mode='same')
def upsample(image, factor=2):
    """
    interpoloate the image with factor 
    """
    width, height = image.shape
    # resulting signal
    image_upsampled = np.zeros((factor*width,
        factor*height),dtype=np.float)

    #copying the initial points
    image_upsampled[::factor,::factor] = image

    #convolving with a kernel
    kernel = np.ones(factor,dtype=np.float)
    kernel = (1/factor)*np.convolve(kernel, kernel)
    kernel = np.outer(kernel,kernel)


    image_upsampled = convolve(image_upsampled, kernel)

    #move back to integer 
    # image_upsampled = np.ceil(255*image_upsampled).astype('int')

    return image_upsampled
def downsample(img,factor):
    """
    Down sample the image by factor
    """

    #  Blur or not: A.Belcaid # 


    down = img[::factor,::factor].copy()

    return down
def rho(scale, sigma_min=0.8, delta_min=0.5,n_spo=3):
    """Function to compute the standard deviation of the next scale in the
    pyramid (Check anatomy of the Sift method page 375)
    

    :scale: current scale (s) in the article
    :sigma_min: minimal standard deviaiton
    :delta_min: minitla inter pixel distance
    :n_spo    : number of scales per octave
    :returns: sigma on scale (s)
    """
    diff = 2**(2*scale/n_spo) - 2**(2*(scale-1)/n_spo)
    return sigma_min/delta_min * np.sqrt(diff)
def first_octave(U_in, delta_min=0.5, sigma_min=0.8, sigma_in=0.5, n_spo=3):
    """Function to generate the first pyramid by upsampling with an inner
    distance (delta_min)

    :U_in: initial image with size (M, N)
    :delta_min: Inter pixel distance =0.5 means upsampling by a factor of 2
    :sigma_min: Minimal blurring value
    : sigma_in: initial blur standard deviation
    :n_spo    : number of scales per octave
    :returns: Tensor of images (M',N', n_spo + 2)
    : Sigmas: list of sigmas for each octave


    """

    ##Upsampling the image
    factor = int(1/delta_min)
    v_0 = upsample(U_in, factor=factor)

    #getting the shape
    N, M = v_0.shape
    Pyramid = np.zeros((n_spo+3, N, M ),dtype=v_0.dtype)
    
    #list of sigmas
    Sigmas = np.zeros(n_spo+3)

    #first image is obtained by upsampling followed by a convolution with a
    #gaussian
    sigma = np.sqrt(sigma_min**2 - sigma_in**2)/delta_min
    Pyramid[0] = gaussian(v_0,sigma)
    Sigmas[0] = sigma_min


    #recursive loop to compute the other scales
    for s in range(1,n_spo+3):
        #computing the associated sigma
        sigma = rho(s, sigma_min, delta_min, n_spo)

        #Blurring the previous image in the octave with sigma
        Pyramid[s] = gaussian(Pyramid[s-1], sigma)

        #saving the associated sigma
        Sigmas[s] = np.sqrt(Sigmas[s-1]**2 + sigma**2)

    #Clipping to 0,1 values
    # Pyramid = np.clip(Pyramid, 0, 1)

    return Pyramid, Sigmas
def next_octave(U_in, sigma_in, delta_min=0.5, sigma_min=0.8, n_spo=3):
    """ 
    Compute the next octave from the a seed image U_in and sigma_in
    """

    #subsamplig by 2
    U = U_in[::2, ::2]

    M, N  = U.shape

    #octave as a tensor
    Octave = np.zeros((n_spo+ 3, M, N), dtype=U.dtype)

    #list of sigmas
    Sigmas = np.zeros(n_spo+3)

    #first image
    Octave[0] = U
    Sigmas[0] = sigma_in

    #recursive filling the octaves
    for s in range(1,n_spo+3):
        sigma = rho(s, sigma_min, delta_min,n_spo)
        Octave[s] = gaussian(Octave[s-1], sigma)
        Sigmas[s] = np.sqrt(Sigmas[s-1]**2 + sigma**2)
    
    return Octave, Sigmas
def _normalize(Array):
    """
    Function to normalize an array for better visualisation
    """

    #extract the minimal and maximal values
    min_value, max_value = Array.min(), Array.max()


    return (Array - min_value) / (max_value - min_value)
def gaussian_pyramid(U_in, delta_min=0.5,sigma_in=0.5, sigma_min=0.8, n_spo=3,n_oct=8):
    """
    Function to compute the Gaussian pyramid using parameters
    :U_in:         Initial image
    :delta_min:    Minimal inter pixel distance
    :sigma_in :     initial sigma
    :sigma_min:   minimal blurring standard devition sigma
    :n_spo:       number of scales per octave
    : n_oct: number of octaves to generate
    """

    
    # First octave
    assert( n_oct > 0), ' number of octaves should be non nul'
    octave, sigmas= first_octave(U_in,delta_min, sigma_min, sigma_in, n_spo)
    yield octave, sigmas


    #input image for the next octave
    U_in     =  octave[n_spo]
    sigma_in =  sigmas[n_spo]

    for i in range(1,n_oct):
        #computing the next octave
        octave, sigmas = next_octave(U_in, sigma_in,delta_min, sigma_min, n_spo)
        yield octave, sigmas

        #prepare the next octave
        U_in  = octave[n_spo]
        sigma_in = sigmas[n_spo]
def show_octave(octave, sigmas=None, candidates=None,octave_index=None):
    """Function to plot an octave with matplotlib

    :octave: octave ( 4 or 3 dimensionnal images depending on the number of
    channels (n_spo, M, N, c): where n_spo are the number of images in the
    octave, (M,N) are the widths of the image and c is the number of channels
    :sigmas:  Associated sigmas for each image
    :candiates: A dictionnary retaining the possible candidates points

    :returns: None
    """

    #  TODO: Add a new support for the new candidates format# 

    #number of scales
    n_spo = octave.shape[0]
    
    fig, axs =  plt.subplots(1, n_spo, figsize = (4*n_spo, n_spo))

    # add circles in case of sigmas
    if candidates:
        # for s in range(1,n_spo-1):
        #     for point in candidates[(octave_index,s)]:
        #         circle(octave[s],point,10,color=0,thickness=3)
        for (o, s, m, n) in candidates:
            if o == octave_index:
                circle(octave[s], (m,n), 10, color=0.5, thickness=3)


    for i in range(n_spo):
        axs[i].imshow(_normalize(octave[i]))
        axs[i].axis('off')

    #titles
    if sigmas is not None:
        for ax,sigma in zip(axs,sigmas):
            ax.set_title(r'$\sigma$= {:.2f}'.format(sigma), fontsize= 16)
def DoG(Octave):
    """Compute the difference of gaussian in the current gaussian pyramid octave

    :Octave: Set of sigma scales images (s=0,...,n_spo+2)
    :returns: Set of differences (s=0,....,n_spo+1)

    """
    return Octave[1:,:,:] - Octave[:-1, :, :]
def is_Pyramid_peak(octave ,s, m,n):
    """
    Verify if the point (octave, s, m, n) is a local maxima
    """


    #comparing to all values of the in the previous octave
    if not np.all(octave[s,m,n] >= octave [s-1, m-1:m+2, n-1:n+2]):
        return False

    #comparing to all values of the in the next value octave
    if not np.all(octave[s,m,n] >= octave [s+1, m-1:m+2, n-1:n+2]):
        return False


    #comparing to all values of the in the same value octave
    if not np.all(octave[s,m,n] >= octave [s, m-1:m+2, n-1:n+2]):
        return False

    return True
def local_extremums(Pyramid):
    """Find a list of local maxima in the Pyramid
    : dog_Pyramid: Pyramid of the Difference of Gaussian
    :returns:  a list of tuples in the form (o, s, m, n)
    o:octave, s:scale, (m,n): image position
    """
    
    #list of condidates
    # candidates = defaultdict(lambda: []) # default value is empty list
    candidates = []
    
    #loop on each scale
    for o,octave in enumerate(Pyramid):

        #loop on scales        
        n_spo = octave.shape[0]-2
        M, N  = octave.shape[1], octave.shape[2]
        print("checking octave ", o)

        #loop on each scale
        for s in range(1,n_spo+1):
            #loop on image dimensions
            for m in range(1,M-1):
                for n in range(1,N-1):
                    if is_Pyramid_peak(octave, s , m, n):
                        #add the point (m,n) to index (o,s)
                        # candidates[(o,s)].append((m, n))
                        candidates.append((o,s,m,n))

    return candidates
def discard_low_contrast(Dog, candidates, c_dog=0.015):
    """
    Function to discard low contrast candidates that are sensible to noise
    
    :Dog: Difference of gaussian
    : candidates: candidates for the local extrema
    :d_dog: threshold on the values
    """

    n_spo = Dog[0].shape[0]             # each octave is a tensor 
    D = [np.abs(o) for o in Dog]
    
    n_oct = len(Dog)
    new_condidates = []

    for o,s,m,n in candidates:
        if D[o][s, m, n] >= 0.8* c_dog:
            new_condidates.append((o,s,m,n))

    return new_condidates
def gradient(DoG, s, m, n):
    """
    Function to compute the discrete gradient in the point P(s, m, n).
    The scheme for each partial derivative is centered

    :DoG: tensor indexed  by (s, m, n)
    :grad:  [ds(P), dm(P), dn(P) ] ^T
    """

    ds = ( DoG[s+1,m,n] - DoG[s-1, m, n] ) /2
    dm = ( DoG[s, m+1 ,n] - DoG[s, m-1 , n] ) /2
    dn = ( DoG[s,m,n+1] - DoG[s, m, n-1] ) /2

    return np.array([ds, dm, dn])
def Hessian(DoG, s, m, n):
    """Function to compute the Hessian with finite difference at the point
    P(s, m, n)

    :DoG:  Difference of Gaussian at a scale o shape (n_spo,M, N)
    :s:    Current sigma scale
    :m:    Height position in the image 
    :n:    Width position in the image
    :returns: Value of the hessian (matrice (3 by 3) )

    """
    
    #compute second derivative
    dss = ( DoG[s-1, m, n ] -2*DoG[s, m, n] + DoG[s+1, m, n] )
    dmm = ( DoG[s, m-1, n ] -2*DoG[s, m, n] + DoG[s, m+1, n] )
    dnn = ( DoG[s, m, n-1 ] -2*DoG[s, m, n] + DoG[s, m, n+1] )

    #combined derivative
    dsm =(DoG[s+1,m+1,n]-DoG[s+1,m-1,n]-DoG[s-1,m+1,n]+DoG[s-1,m-1,n])/4
    dsn =(DoG[s+1,m,n+1]-DoG[s+1,m,n-1]-DoG[s-1,m,n+1]+DoG[s-1,m,n-1])/4
    dmn =(DoG[s,m+1,n+1]-DoG[s,m+1,n-1]-DoG[s,m-1,n+1]+DoG[s,m-1,n-1])/4

    return np.array([[dss, dsm, dsn], [dsm, dmm, dmn], [dsn, dmn, dnn]])
def quadratic_interpolation(DoG, s, m,n):
    """Compute the Dog continuous extrema by solving the equation
     Hessian * alpah = -gradient

    :DoG: Tensor of size (n_spo, M, N)
    :s:    scale position
    :m:    height position
    :n:    width position
    :returns:
    :alpha:the solution (alpha_s, alpha_m, alpha_n)
    :omega: interpolated extremum
    """

    #compute alpha
    H = Hessian(DoG, s, m, n)
    g = gradient(DoG,s, m, n)
    alpha = np.linalg.solve(H, -g)
    
    #compute omega
    omega = DoG[s,m,n]  +0.5* np.dot(alpha,g)

    return alpha, omega
def keypoints_interpolation(DoG, candidates,max_repeat=5,sigma_min=0.8):
    """Main function to filter the key candidates by interpolation
    :DoG:  List of tensor for each octave. Each octave is a tensor of size
    (N_spo, M, N)
    :candidates: A list of tuples (o, s , m, n) (octave, scale, (m,n) position)
    :returns:  New set of candidates (o_e, s, m, n, sigma, x, y, w)
    """

    #initial features
    features = []
    #main loop on candidates
    for (o,s,m,n) in candidates:
        #initial boolean variable to pass
        is_interior = False #max(alpha) < 0.6
        curr_iter   = 0     #current iteration
        # print("initial position ({}, {},{}))".format(s,m,n))
        S,M, N = DoG[o].shape
        while (not is_interior) and curr_iter < max_repeat:
            #computethe gradient and image by interpolation
            alpha,  w  = quadratic_interpolation(DoG[o], s, m, n)

            #compute corresponding absolute candidates
            #  TODO: compute absolute candidate # 
            sigma = 2**o*sigma_min*2**(alpha[0]+s)/S
            x     = 2**(o-1)*(alpha[1]+m)
            y     = 2**(o-1)*(alpha[2]+n)

            #update the interpolation position
            s, m, n = np.floor([s+alpha[0],m + alpha[1], n +
                alpha[2]]).astype('int')
            # print("new position ({}, {},{}))".format(s,m,n))

            #check for validity otherwise throw away
            if(s<1 or s>=S-1 or m<1 or m>=M-1 or n<1 or n>=N-1):
                break

            #update is_interior
            #  TODO: Get the correct validation # 
            if np.max(alpha) <0.6:
                is_interior =  True
                features.append((o,s,m,n,sigma,x,y,w))

            #update current iteration
            curr_iter += 1

    return features
def discard_low_constrasted_candidates(candidates, C_dog=0.015):
    """ Discart low contrast candidates based on a given threshold.

    :candidates: List [(o,s,m,n, sigma, x, y, w)]
    :C_dog: threshold for the gradient
    :returns: new list [(o,s,m,n, sigma, x, y, w)] if |W|>=Cdog
    """
    
    return [candidat for candidat in candidates\
            if abs(candidat[-1])>=C_dog]
def discard_edges_candidates(DoG, candidates, C_edge=10):
    """
    Function to discard key candidate on edges by comparing eigen values of
    the hessian
    Inputs:
    :DoG:  Difference of Gaussian space
    :candidates: (o,s,m,n,sigma, x, y, w) list
    :Outputs:
    : candidates: (o, s, m, n, sigma, x, y , w) new list
    """

    new_candidates = []

    threshold = (C_edge +1)**2 / C_edge

    for o, s, m, n, sigma, x, y, w in candidates:

        #compute the hessian 
        H = Hessian(DoG[o],s, m, n)

        #compute edgness
        edgness = np.trace(H)**2 / np.linalg.det(H)

        if edgness < threshold:
            new_candidates.append((o,s,m,n,sigma, x, y, w))

    return new_candidates
class SIFT(object):

    """Class to perform Sift feature extractor on a given imag"""

    def __init__(self, img,delta_min=0.5,sigma_in=0.5, sigma_min=0.8,\
            n_spo=3,n_oct=8,lambda_ori=1.5,lambda_desc=6,nbins=128):
        """Constructor with color or gray scale image

        :img: TODO

        """
        #store initial image 
        self.initial_img = img
        self.delta_min = delta_min     #minimal inter distance
        self.sigma_in  = sigma_in      #initial bluring sigma
        self.sigma_min = sigma_min     # minimal sigma 
        self.n_spo     = n_spo         # number of scales per octave
        self.n_oct     = n_oct         # number of octave
        self.lambda_ori = lambda_ori   # scaling factor for the local window
        self.nbins      = nbins        # number of bins in a keypoint
        self.lambda_desc = lambda_desc



        # if 3 channels colors make it gray scale
        if(img.ndim == 3):
            self.img = rgb2gray(img)
        else:
            self.img = img


        ##############
        #  Pyramids  #
        ##############
        #computing the pyarmid
        pyramid = list(self._compute_gaussian_pyramid())
        self.pyramid  = [P[0] for P in pyramid]           #gaussian pyramid
        self.sigmas   = [P[1] for P in pyramid] 
        self.DoG      = [DoG(octave) for octave in self.pyramid]           # difference of gaussian pyramid
        self.gradient = {}           # Gradient of the dog (centreddifference)
        self.candidates = None         # candidates


        ################
        #  candidates  #
        ################
    
    def _compute_gaussian_pyramid(self):
        """
        Function to compute the Gaussian pyramid using parameters
        :U_in:         Initial image
        :delta_min:    Minimal inter pixel distance
        :sigma_in :     initial sigma
        :sigma_min:   minimal blurring standard devition sigma
        :n_spo:       number of scales per octave
        : n_oct: number of octaves to generate
        """

        
        # First octave
        assert( self.n_oct > 0), ' number of octaves should be non nul'
        octave, sigmas= first_octave(self.img,self.delta_min,self.sigma_min,
                self.sigma_in,self.n_spo)
        yield octave, sigmas


        #input image for the next octave
        U_in     =  octave[self.n_spo]
        sigma_in =  sigmas[self.n_spo]

        for i in range(1,self.n_oct):
            #computing the next octave
            octave, sigmas = next_octave(U_in, sigma_in,self.delta_min,
                    self.sigma_min, self.n_spo)
            yield octave, sigmas

            #prepare the next octave
            U_in  = octave[self.n_spo]
            sigma_in = sigmas[self.n_spo]

    def show_octave(self,octave_index, is_dog=False):
        """Function to plot an octave with matplotlib

        :octave: octave ( 4 or 3 dimensionnal images depending on the number of
        channels (n_spo, M, N, c): where n_spo are the number of images in the
        octave, (M,N) are the widths of the image and c is the number of channels
        :sigmas:  Associated sigmas for each image
        :candiates: A dictionnary retaining the possible candidates points

        :returns: None
        """

        
        #preparing the octave
        octave = self.pyramid[octave_index]\
                if not is_dog else self.DoG[octave_index]

        #number of scale in the current octave
        n_spo = octave.shape[0]

        fig, axs =  plt.subplots(1,n_spo, figsize = (4*n_spo, self.n_spo))



        for i in range(n_spo):
            axs[i].imshow((octave[i]))
            axs[i].axis('off')

        #titles
        for ax,sigma in zip(axs,self.sigmas[octave_index]):
                ax.set_title(r'$\sigma$= {:.2f}'.format(sigma), fontsize= 16)
    def local_extremums(self):
        """Find a list of local maxima in the Pyramid
        : dog_Pyramid: Pyramid of the Difference of Gaussian
        :returns:  a list of tuples in the form (o, s, m, n)
        o:octave, s:scale, (m,n): image position
        """
        
        #list of condidates
        candidates = []
        
        #loop on each scale
        for o,octave in enumerate(self.DoG):

            #loop on scales        
            M, N  = octave.shape[1], octave.shape[2]
            print("checking octave ", o)
            print("octave size is {}".format((M,N)))

            #loop on each scale
            for s in range(1,self.n_spo+1):
                #loop on image dimensions
                for m in range(1,M-1):
                    for n in range(1,N-1):
                        if is_Pyramid_peak(octave, s , m, n):
                            candidates.append((o,s,m,n))

        return candidates
    def _candidates_to_cv_keys(self,candidates):
        """
        Function to convert a set of detected candidates to opencv Keypoint
        candates are either in the form of (o, s, m, n) or more detailled
        representation
        """

        #keypoint dictionnary
        keys_dict = {}

        for o,s, m,n in candidates:
            
            #extract correct coordinates given octave
            scale = 2**(o-1)
            x, y = int(scale*m), int(scale* n)

            #updating the key
            keys_dict[(x,y)] = s

        return [cv2.KeyPoint(x,y,5*s) for  (x,y), s in keys_dict.items()]
    def _detailled_candidates_to_cv_keys(self,candidates):
        """
        Function to convert a set of detected candidates to opencv Keypoint
        candates are either in the form of (o, s, m, n,sigma,x,y,w) or more detailled
        representation
        """

        #keypoint dictionnary
        keys_dict = defaultdict(int)

        for o,s, m,n,sigma, x,y,w in candidates:
            
            #extract correct coordinates given octave
            x, y = int(x), int(y)

            #updating the key
            keys_dict[(x,y)] = max(keys_dict[(x,y)],  5*sigma)

        return [cv2.KeyPoint(x,y,sigma) for  (x,y), sigma in keys_dict.items()]
    def _oriented_candidates_to_cv_keys(self,candidates):
        """
        Function to convert a set of detected candidates to opencv Keypoint
        candates are either in the form of (o, s, m, n,sigma,x,y,w,theta) or more detailled
        representation
        """

        #keypoint dictionnary
        keys_dict = defaultdict(int)

        for o,s, m,n,sigma, x,y,w, theta in candidates:
            
            #extract correct coordinates given octave
            x, y = int(x), int(y)

            #updating the key
            keys_dict[(x,y,theta)] = max(keys_dict[(x,y,theta)],  5*sigma)

        return [cv2.KeyPoint(x,y,sigma,(theta+np.pi)*(360/2*np.pi)) for\
                (x,y,theta), sigma in keys_dict.items()]
    def detect(self):
        """Compute the keypoints for the current image. 
        """

        #computing the local maximas
        maximas = self.local_extremums()


        

        ##filtering low contrast
        maximas= discard_low_contrast(self.DoG,maximas)
        


        ##interpoloation
        self.candidates = keypoints_interpolation(self.DoG,maximas)

        ##removing edges points

        self.candidates = discard_edges_candidates(self.DoG,self.candidates)
    def get_keypoints(self):
        """
        Function to return the internal list of keypoints in opencv format 
        """
        
        if self.candidates:
            return self._detailled_candidates_to_cv_keys(self.candidates)

        else:
            return None
    def _get_patch(self, keypoint):
        """
        compute the the local pathc with radious self.lambda_ori around the
        keypoint
        """
        
        #element of the keypoint
        o,s, m, n , sigma, x, y, w = keypoint

        h,w = self.initial_img.shape[0], self.initial_img.shape[1]

        #cherck that the patch in inside the initial image
        if ( x < 3*self.lambda_ori*sigma or x> h - 3*self.lambda_ori *sigma):
            return None
        if ( y < 3*self.lambda_ori*sigma or y> w - 3*self.lambda_ori *sigma):
            return None

        #interpixel for octave o
        delta_o = 2**(o-1)

        #limits
        x_min, x_max = int((x-3*self.lambda_ori*sigma)/delta_o), int((x+3*self.lambda_ori*sigma)/delta_o)
        y_min, y_max = int((y-3*self.lambda_ori*sigma)/delta_o), int((y+3*self.lambda_ori*sigma)/delta_o)


        return [(x_min,x_max), (y_min,y_max)]
    def orientation_accumulation(self):
        """
        Function to perform the first step in the local analysis to 
        extract the main orientation in a key point 
        """
        
        #ist of oriented candiates
        oriented_candidates = []
        
        #smoothing kernel
        kernel = np.ones(3)/3


        #delta theta (angle for each bin)
        delta_theta = 2*np.pi/self.nbins

        for keypoint in self.candidates:
            o,s,m,n,sigma,x,y,w  = keypoint

            #interpixel
            delta_o = 2**(o-1)
            

            #computing the patch
            patch =self._get_patch(keypoint)



            if patch is not None:

                #getting the sise of the patch
                M1, M2  = patch[0]
                N1, N2  = patch[1]

                #bins patch 
                h = np.zeros(self.nbins)

                for m in range(M1, M2):
                    for n in range(N1, N2):
                        #computing the Gradient
                        G = gradient(self.DoG[o], s,m,n)
                        G_norm = np.linalg.norm(G[1:]) #grad (\delta_m, _n)
                        
                        
                        #diff
                        diff = np.array([delta_o*m-x, delta_o*n - y])
                        diff = np.sum(diff**2)


                        #coeffcient 
                        c_ori = np.exp(- diff/(2*(self.lambda_ori*sigma)**2))\
                                *G_norm

                        #corresponding index
                        angle = np.arctan2(G[1], G[2])
                        index = int(self.nbins/(2*np.pi) * (angle % 2*np.pi) )
                        h[index]+= c_ori

                # smooth the histogram
                for _ in range(6):
                    h = np.convolve(h, kernel, mode='same')

                # computing the main reference angles
                threshold  = 0.8*np.max(h)

                for i  in range(self.nbins):
                    #left neighbor
                    h_minus = h[(i-1) % self.nbins]

                    #right neighbor
                    h_plus = h[(i+1) % self.nbins]

                    h_i =  h[i]

                    if (h_i > h_minus) and (h_i > h_plus) and (h_i >=threshold):
                        #theta_k 
                        theta_k = (i+0.5) * delta_theta  

                        #theta_ref
                        theta_ref = theta_k + np.pi/self.nbins*\
                                (h_minus - h_plus) / (h_minus - 2*h_i + h_plus)
                        

                        oriented_candidates.append((o,s,m,n,sigma,x,y,w,theta_ref))


            self.candidates = oriented_candidates
                    
    def _patch_within(self,x,y, sigma):
        """"
        Check if we could construct a patch within the image
        The patrch radius is sqrt(2)*lambda_desc*sigma
        """

        #radius of the patch
        radius = int( np.sqrt(2) * self.lambda_desc * sigma )

        #shape of the image
        h, w = self.initial_img.shape[:2]

        #check validity
        if( x <radius or x > h - radius):
            return False
        
        if( y < radius or y > w - radius):
            return False
        
        return True

    def describe(self):
        """
        Function to extract the local feature from the oriented keypoints 

        :input: keypoints a set of points in the form (o,s, m, n, sigma,
        x, y, w, theta_ref)

        :output: feature vector for each point 
        (o,s,m,n,x,y,w,theta_ref, f)
        """

        #initial image width
        h, w = self.initial_img.shape[:2]

        for keypoint in self.candidates:

            #extracting the elements of the keypoint
            o,s,m,n,sigma,x,y,w,theta = keypoint
            print(keypoint)

            #inter pixel
            delta_o = 2**(o-1)

            if self._patch_within(x, y, sigma):

                #array of weighted histograms
                h = np.zeros((4,4,8))
                print("inside")

                #limits 
                m_inf = int((x - np.sqrt(2)*sigma*5/4)/delta_o)
                m_max = int((x + np.sqrt(2)*sigma*5/4)/delta_o)
                n_inf = int((y - np.sqrt(2)*sigma*5/4)/delta_o)
                n_max = int((y + np.sqrt(2)*sigma*5/4)/delta_o)

                # Accumulate the keypoint in the histogram
                for m in range(m_inf, m_max+1):
                    for n in range(n_inf, n_max+1):

                        #compute rotated normalized coordinates
                        x_hat  =  ((m*delta_o - x)* cos(theta) + (n*delta_o - y)
                                * sin(tetha))/ sigma

                        y_hat  =  (-(m*delta_o - x) *sin(theta) + (n*delta_o
                            -y)*cos(theta))/dsigma
                        print('x hat is {}, y hat is {}'.format(x_hat, y_hat))

                


                

                















        


