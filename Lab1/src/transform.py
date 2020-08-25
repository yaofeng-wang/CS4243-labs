import numpy as np
from skimage import io
import os.path as osp

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = np.asarray(io.imread(file_name))
    if len(image.shape)==3 and image.shape[2]>3:
        image = image[:, :, :3]
    # print(image.shape) #should be (x, x, 3)
    return image

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def cs4243_resize(image, new_width, new_height):
    """
    5 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    Please round down the value to its nearest interger, 
    and take care of the order of image dimension.
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    ###Your code here###
    
    # if new_width < 0 or new_height < 0, np.zeros() will throw a ValueError.
    
    # if new_width == 0 or new_height == 0, we won't need to do any calculation.
    if new_width == 0 or new_height == 0:
        return new_image
    
    # resizing algorithm taken from
    # https://tech-algorithm.com/articles/nearest-neighbor-image-scaling/
    height, width = image.shape[0], image.shape[1]
    w_ratio = int(((width << 16) / new_width) + 1)
    h_ratio = int(((height << 16) / new_height) + 1)
    
    for h in range(new_height):
        for w in range(new_width):
            pw = int((w*w_ratio) >> 16)
            ph = int((h*h_ratio) >> 16)
            new_image[h,w] = image[ph, pw]
    ###
    return new_image    
    

def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function, use the
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    ###Your code here####
    # construct weights numpy
    weights = np.array([0.299, 0.587, 0.114])
    
    # multiply pixel RGB with weights and sum up the RGB axis
    image = np.dot(image, weights)
    
    ###

    return image/255.

def cs4243_histnorm(image, grey_level=256):
    """
    5 points 
    Stretch the intensity value to [0, 255]
    :param image : ndarray
    :param grey_level
    :return res_image: hist-normed image
    Tips: use linear normalization here https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    res_image = image.copy()
    ##your code here ###
    min_pixel = np.amin(res_image)
    max_pixel = np.amax(res_image)
    res_image = (res_image - min_pixel) / (max_pixel - min_pixel) * (grey_level-1)
    ####
    return res_image



def cs4243_histequ(image, grey_level=256):
    """
    10 points
    Apply histogram equalization to enhance the image.
    the cumulative histogram will aso be returned and used in the subsequent histogram matching function.
    :param image: numpy.ndarray(float64)
    :return: ori_hist: histogram of original image
    :return: cum_hist: cumulated hist of original image, pls normalize it with image size.
    :return: res_image: image after being applied histogram equalization.
    :return: uni_hist: histogram of the enhanced image.
    Tips: use numpy buildin funcs to ease your work on image statistics
    """
    ###your code here####
    
    # get the original histogram
    x, y = image.shape
    hist = [0] * grey_level
    for i in range(x):
        for j in range(y):
            hist[image[i, j]]+=1
    ori_hist = hist
    
    # get the cumulative distribution function (CDF) normalised to image size
    cum_hist = [sum(ori_hist[:i+1]) for i in range(len(ori_hist))]
    cum_hist = np.array(cum_hist) / (x*y)
    
    # get the uniform histogram from normalised CDF
    uniform_hist = np.uint8((grey_level-1) * cum_hist)
    
    ###

    # Set the intensity of the pixel in the raw image to its corresponding new intensity 
    height, width = image.shape
    res_image = np.zeros(image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = uniform_hist[image[i,j]]
    
    uni_hist = np.bincount(res_image.flatten(), minlength=grey_level)
    return ori_hist, cum_hist, res_image, uni_hist
 
def cs4243_histmatch(ori_image, refer_image):
    """
    10 points
    Map value according to the difference between cumulative histogram.
    Note that the cum_hists of the two images can be very different. It is possible
    that a given value cum_hist[i] != cum_hist[j] for all j in [0,255]. In this case, please
    map to the closest value instead. if there are multiple intensities meet the requirement,
    choose the smallest one.
    :param ori_image #image to be processed
    :param refer_image #image of target gray histogram 
    :return: ori_hist: histogram of original image
    :return: ref_hist: histogram of reference image
    :return: res_image: image after being applied histogram matching.
    :return: res_hist: histogram of the enhanced image.
    Tips: use cs4243_histequ to help you
    """
    
    ##your code here ###
    
    ##
    # Set the intensity of the pixel in the raw image to its corresponding new intensity      
    height, width = ori_image.shape
    res_image = np.zeros(ori_image.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            res_image[i,j] = map_value[ori_image[i,j]]
    
    res_hist = np.bincount(res_image.flatten(), minlength=256)
    
    return ori_hist, ref_hist, res_image, res_hist


def cs4243_rotate180(kernel):
    """
    Rotate the matrix by 180. 
    Can utilize build-in Funcs in numpy to ease your work
    :param kernel:
    :return:
    """
    kernel = np.flip(np.flip(kernel, 0),1)
    return kernel

def cs4243_gaussian_kernel(ksize, sigma):
    """
    5 points
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize))
    ###Your code here####
    cp = (int)(ksize/2)
    for i in range(ksize):
        for j in range(ksize):
            kernel[i,j] = np.exp(((i-cp)**2 + (j-cp)**2) / (-2*sigma**2))
    ###
    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    10 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    # pad image to handle border pixels
    pad_height = (int)((Hk - 1)/2)
    pad_width = (int)((Wk - 1)/2)
    image_pad = pad_zeros(image, pad_height, pad_width)

    # Flip the kernel horizontal and vertical
    kernel = cs4243_rotate180(kernel)
    
    # compute effective output size, assume stride=1
    out_height = 1 + Hi - Hk + 2*pad_height
    out_width = 1 + Wi - Wk + 2*pad_width
    
    # get initial nodes of receptive fields
    recep_fields_h = [i for i in range(out_height)]
    recep_fields_w = [i for i in range(out_width)]
    
    for i in recep_fields_h:
        for j in recep_fields_w:         
            # get receptive area
            recep_area = image_pad[i:i+Hk, j:j+Wk]       

            # multiply recep_area with kernel
            conv_sum = 0.0
            for y in range(Hk):
                for x in range(Wk):                    
                    conv_sum += kernel[y][x] * recep_area[y][x]
            filtered_image[i, j] = conv_sum
    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    10 points
    Implement a fast version of filtering algorithm.
    take advantage of matrix operation in python to replace the 
    inner 2-nested for loops in filter function.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    
    # pad image to handle border pixels
    pad_height = (int)((Hk - 1)/2)
    pad_width = (int)((Wk - 1)/2)
    image_pad = pad_zeros(image, pad_height, pad_width)

    # Flip the kernel horizontal and vertical
    kernel = cs4243_rotate180(kernel)
    
    # compute effective output size, assume stride=1
    out_height = 1 + Hi - Hk + 2*pad_height
    out_width = 1 + Wi - Wk + 2*pad_width
    
    # get initial nodes of receptive fields
    recep_fields_h = [i for i in range(out_height)]
    recep_fields_w = [i for i in range(out_width)]
    
    for i in recep_fields_h:
        for j in recep_fields_w:         
            # get receptive area
            recep_area = image_pad[i:i+Hk, j:j+Wk]     
            filtered_image[i, j] = np.multiply(kernel, recep_area).sum()
    ###

    return filtered_image

def cs4243_filter_faster(image, kernel):
    """
    10 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk),also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and rehshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    Tips: You may find the functions pad_zeros() and cs4243_rotate180() useful
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    
    # pad image to handle border pixels
    pad_height = (int)((Hk - 1)/2)
    pad_width = (int)((Wk - 1)/2)
    image_pad = pad_zeros(image, pad_height, pad_width)
    
    # compute effective output size, assume stride=1
    out_height = 1 + Hi - Hk + 2*pad_height
    out_width = 1 + Wi - Wk + 2*pad_width
    
    # get initial nodes of receptive fields
    recep_fields_h = [i for i in range(out_height)]
    recep_fields_w = [i for i in range(out_width)]
    
    # extract receptive area into matrix of shape (Hi*Wi, Hk*Wk)
    recep_areas = []
    for i in recep_fields_h:
        for j in recep_fields_w:
            recep_areas.append(image_pad[i: i+Hk, j: j+Wk].reshape(-1))
    out = np.stack(recep_areas)
    
    # Flip the kernel horizontal and vertical
    kernel = cs4243_rotate180(kernel).reshape(Hk*Wk, 1)
    
    # dot product kernel and receptive areas
    filtered_image = np.dot(out, kernel).reshape(Hi, Wi)
    
    ###

    return filtered_image

def cs4243_downsample(image, ratio):
    """
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """
    width, height = image.shape[1], image.shape[0]
    return image[0:height:ratio, 0:width:ratio]

def cs4243_upsample(image, ratio):
    """
    upsample the image to its 2^ratio, 
    :param image: image to be upsampled
    :param kernel: use same kernel to get approximate value for additional pixels
    :param ratio: which means upsample the width to ratio*width, and height to ratio*height
    :return res_image: upsampled image
    """
    width, height = image.shape[1], image.shape[0]
    new_width, new_height = width*ratio, height*ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = image
    return res_image


def cs4243_gauss_pyramid(image, n=4): 
    """
    10 points
    build a Gaussian Pyramid of level n
    :param image: original grey scaled image
    :param n: level of pyramid
    :return pyramid: list, with list[0] corresponding to original image.
	:e.g., img0->blur&downsample->img1->blur&downsample->img2	
    Tips: you may need to call cs4243_gaussian_kernel() and cs4243_filter_faster()
	The kernel for blur is given, do not change it.
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    pyramid = []
    ## your code here####

    pyramid = [image]
    for i in range(n):
        gpyr_image = cs4243_filter_faster(pyramid[i], kernel)
        gpyr_image = cs4243_downsample(gpyr_image, 2)
        pyramid.append(gpyr_image)
    
    ##
    return pyramid

def cs4243_lap_pyramid(gauss_pyramid):
    """
    10 points
    build a Laplacian Pyramid from the corresponding Gaussian Pyramid
    :param gauss_pyramid: list, results of cs4243_gauss_pyramid
    :return lap_pyramid: list, with list[0] corresponding to image at level n-1 in Gaussian Pyramid.
	Tips: The kernel for blurring during upsampling is given, you need to scale its value following the standard pipeline in laplacian pyramid.
    """
    #use same Gaussian kernel 

    kernel = cs4243_gaussian_kernel(7, 1)
    n = len(gauss_pyramid)
    lap_pyramid = [gauss_pyramid[n-1]] # the top layer is same as Gaussian Pyramid
    ## your code here####
    
    for i in range(n-1, 0, -1):
        upsampled_image = cs4243_upsample(gauss_pyramid[i], 2)
        expanded_image = (cs4243_filter_faster(upsampled_image, kernel)) * 4
        residual = gauss_pyramid[i-1] - expanded_image
        lap_pyramid.append(residual)      
    
    ##
    
    return lap_pyramid
    
def cs4243_Lap_blend(A, B, mask):
    """
    10 points
    blend image with Laplacian pyramid
    :param A: image on left
    :param B: image on right
    :param mask: mask [0, 1]
    :return blended_image: same size as input image
    Tips: use cs4243_gauss_pyramid() & cs4243_lap_pyramid() to help you
    """
    kernel = cs4243_gaussian_kernel(7, 1)
    blended_image = None
    ## your code here####
    
    ##
    
    return blended_image
