import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
import math

from utils import pad, get_output_space, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def warp_image(src, dst, h_matrix):
    dst = dst.copy()
    dst = cv2.warpPerspective(dst, np.linalg.inv(h_matrix), (src.shape[1] + dst.shape[1], src.shape[0]))
    dst[0:src.shape[0], 0:src.shape[1]] = src
    return dst

def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas

def transform_homography(src, h_matrix, getNormalized = True):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    """
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """
    h_matrix = np.eye(3, dtype=np.float64)

    ### YOUR CODE HERE
       
    '''
    0. convert x and x' to homogenous
    1. normalize x
    2. normalize x'
    3. DLT
    4. combine 1, 2, 3
    
    DLT
    1. For each correspondence compute A_i
    2. stack the A_i into A. A has shape 2n x 9
    3. Get SVD of A.
    4. H is the column with smallest singular value
    
    '''
    # this function computes a mapping from src to dst,
    # instead of from src to dst as per the instructions given
    
    
    N = len(src)
    
    if N < 4:
        # error
        pass
    
    # covert to homogeneous matrix
    src = pad(src)
    dst = pad(dst)
    
    # normalize src and dst
    T_src, norm_src = normalize(src)
    T_dst, norm_dst = normalize(dst)
    
    # compute A
    l = []
    for i in range(N):
        l.append(get_Ai(norm_src[i], norm_dst[i]))
    A = np.concatenate(l, axis=0)
    
    # get SVD of A
    _, _, vh = np.linalg.svd(A)
    
    # get H
    H = vh[-1].reshape((3,3))
    
    # combine results
    h_matrix = np.dot(np.dot(np.linalg.inv(T_dst), H), T_src)
    h_matrix = h_matrix / h_matrix[2][2] # not sure why, last value should be 1
    ### END YOUR CODE

    return h_matrix

def get_Ai(src_kp, dst_kp):
    
    x = src_kp[0]
    y = src_kp[1]
    x_p = dst_kp[0]
    y_p = dst_kp[1]
    
    Ai = np.array([
            [-x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p],
            [0, 0, 0, -x, -y, -1, x*y_p, y*y_p, y_p]
    ])
    return Ai

def normalize(img):

    m = np.mean(img, axis=0)
    s = np.sqrt(2) / np.mean(euclidean_dist(img - m))
    Tr = np.array([[s, 0, -m[0]*s], 
                  [0, s, -m[1]*s],
                  [0, 0, 1      ]]) # (3 x 3)
    norm_img = np.dot(Tr, img.T)
    return Tr, norm_img.T
    
def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    ### YOUR CODE HERE
    Ix = filters.sobel_v(img)
    Iy = filters.sobel_h(img)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    M_Ixy = convolve(Ixy, window, mode='constant', cval=0.0)
    M_Ixx = convolve(Ixx, window, mode='constant', cval=0.0)
    M_Iyy = convolve(Iyy, window, mode='constant', cval=0.0)
    
    for h in range(H):
        for w in range(W):
            det = M_Ixx[h][w] * M_Iyy[h][w] - M_Ixy[h][w] ** 2
            trace = M_Ixx[h][w] + M_Iyy[h][w]
            
            response[h][w] = det - k * trace ** 2
            
    
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    
    mean = np.mean(patch)
    sd = np.std(patch)
    
    if sd == 0:
        feature = (patch - mean).flatten()
    else:
        feature = ((patch - mean) / sd).flatten()
        
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    M = desc2.shape[0]
    
    for index in range(N):
        dists_sorted = sorted(dists[index])
        closest_dist = dists_sorted[0]
        second_closest_dist = dists_sorted[1]
        
        # did not consider edge cases because the generated matches already looks very
        # similar to the expected outputs, e.g. if descriptor x in desc1 has
        # a closest match descriptor y in desc2 that has a closest match 
        # descriptor z in desc1 such that z != x.
        # 
        if (closest_dist * 1.0 / second_closest_dist) < threshold:
            for m in range(M):
                if dists[index][m] == closest_dist:
                    matches.append([index, m])

    matches = np.array(matches)    
        
    ### END YOUR CODE
    
    return matches

def ransac(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)
    
    # Please note that coordinates are in the format (y, x)
    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    
    matched1[:, [0,1]] = matched1[:, [1,0]]
    matched2[:, [0,1]] = matched2[:, [1,0]]
    matched1_unpad[:, [0,1]] = matched1_unpad[:, [1,0]]
    matched2_unpad[:, [0,1]] = matched2_unpad[:, [1,0]]
    
    for i in range(n_iters):
        indices = np.random.choice(range(N), size=n_samples)
        src = matched1_unpad[indices]
        dst = matched2_unpad[indices]
        
        H = compute_homography(src, dst)
        matched1_trans = transform_homography(matched1_unpad, H)
        
        dist = np.sum((matched1_trans - matched2_unpad) ** 2, axis=1)
        if np.sum(dist < threshold) > n_inliers:
            max_inliers = dist < threshold
            n_inliers = np.sum(dist < threshold)
        
    src = matched1_unpad[max_inliers]
    dst = matched2_unpad[max_inliers]
    H = compute_homography(src, dst)
    
    matched1_trans = transform_homography(matched1_unpad, H)
    dist = np.sum((matched1_trans - matched2_unpad) ** 2, axis=1)
    max_inliers = dist < threshold
    
    ### END YOUR CODE
    return H, matches[max_inliers]

def euclidean_dist(x):
    return np.sqrt(np.sum(x ** 2, axis=1))


def sift_descriptor(patch):
    """
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """
    
    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4,4,8))
    
    ### YOUR CODE HERE
     
    cells = [[[] for _  in range(4)] for _ in range(4)]

    for x in range(16):
        for y in range(16):
            cell_x = x // 4
            cell_y = y // 4
            m = np.sqrt(dx ** 2 + dy ** 2)
            o = (np.arctan2(dy, dx) * 180 / np.pi) + 180 # add 180 so that range of o is 0 to 360
            o_bin = o // 45
            cells[cell_x][cell_y].append(o_bin)

    feature = np.array([])
    for x in range(4):
        for y in range(4):
            hist, bin_edges = np.histogram(cells[x][y], bins=range(9))
            feature = np.concatenate((feature, hist), axis=0)
            
    length = np.sqrt(np.sum(feature ** 2))
    feature = feature / length
    
    assert len(feature) == 128
    # END YOUR CODE
    
    return feature
