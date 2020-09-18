""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


# Part 1 

def smoothing(img):
    """Smooth image using Guassain filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    sigma = 5.0
    img_smoothed = cv2.GaussianBlur(img, (5,5), sigma)

    """ YOUR CODE ENDS HERE """

    return img_smoothed

def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)
    """ YOUR CODE ENDS HERE """
   
    return lab



# Part 2
def k_means_clustering(data,k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()


    """ YOUR CODE STARTS HERE """
    n_samples, n_features = data.shape
    
    MAX_ITERATIONS = 300
    THRESHOLD = 0
    
    # initialise centers; random pick k points from P as centers    
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    centers = data[random_indices, :]
    data = data.astype(int)
    centers = centers.astype(int)
    
    for i in range(MAX_ITERATIONS):      
        # compute squared l2 norm between each center and all other pixels
        l2_distances = []
        for c in centers:
            l2 = np.linalg.norm(data - c, ord=None, axis=1, keepdims=True) ** 2
            l2_distances.append(l2)
        
        # find assigned cluster based on min L2 dist        
        labels = np.argmin(l2_distances, 0).reshape(-1, 1)
    
        # compute new center
        new_centers = np.zeros_like(centers)
        for idx in range(k):
            # indices of data for cluster idx
            indices = np.where(labels == idx)[0] 
            new_centers[idx] = np.mean(data[indices], axis=0)
        
        # break if meet threshold
        if np.abs(new_centers - centers).sum() <= THRESHOLD:
            break 
            
        centers = new_centers  

    """ YOUR CODE ENDS HERE """


    end =  time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs."% kmeans_runtime)
    return labels, centers



def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    
    # compress all coordinates 
    compressed_data = np.round(data / bin_size)
    
    # get similar pixels idx of uniq_seeds
    seeds = {}
    for u in np.unique(compressed_data, axis=0):
        seeds[tuple(u)] = len(np.where((compressed_data == u).all(axis=1))[0])
        
    # filter by min threshold
    bin_seeds = []
    for s, l in seeds.items():
        if l >= min_bin_freq:
            s = [i * bin_size for i in s]
            bin_seeds.append(s)

        
    """ YOUR CODE ENDS HERE """
    return bin_seeds

def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged

    """ YOUR CODE STARTS HERE """
    num_iter = 0
    n_points = 0
    peak = start_seed
    while (num_iter < max_iter):
        # get neighbour points   
        neighbors = nbrs.radius_neighbors(np.expand_dims(peak, axis=0)) # bandwith? #square neighbour
        indices = neighbors[1][0]
        points = data[indices]
        # shift peak to mean of points
        n_points = len(points)
        new_peak = np.mean(points, axis=0)
        # check convergence
        if np.abs(new_peak - peak).sum() <= stop_thresh:
            break 
        peak = new_peak
        num_iter += 1
        
    peak = tuple(peak)  
    
    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)
    


    """ YOUR CODE STARTS HERE """
    # get all peaks of windows
    peaks = np.array(list(center_intensity_dict.keys()))
    # construct class for peaks
    nbrs_peaks = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(peaks)
    
    centers = set()
    for p in peaks:
        # find peaks within bandwidth
        nb = nbrs_peaks.radius_neighbors(np.expand_dims(p, axis=0))
        indices = nb[1][0]
        # if more than 1 peak
        if len(indices) > 1:
            keys = peaks[indices]
            duplicate_peaks = { tuple(k): center_intensity_dict[tuple(k)] for k in keys }
            max_key = tuple(max(duplicate_peaks, key=duplicate_peaks.get))
        else: 
            max_key = tuple(p)
        centers.add(max_key)

    centers = np.array(list(centers))
    
    # assign points to nearest cluster peak        
    nbrs_centers = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(centers)
    d, labels = nbrs_centers.kneighbors(data)

    """ YOUR CODE ENDS HERE """
    end =  time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs."% kmeans_runtime)
    return labels, centers









#Part 3:

def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """

    # reshape to (H x W, -1) to cater for RGB and greyscale img
    img = img.reshape(img.shape[0] * img.shape[1], -1)
    labels, centers = k_means_clustering(img, k)  

    """ YOUR CODE ENDS HERE """

    return labels,centers


def mean_shift_segmentation(img,b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    
    # reshape to (H x W, -1) to cater for RGB and greyscale img
    img = img.reshape(img.shape[0] * img.shape[1], -1)
    labels, centers = mean_shift_clustering(img, bandwidth=b)  
    
    """ YOUR CODE ENDS HERE """

    return labels, centers














"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """


    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):

    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
