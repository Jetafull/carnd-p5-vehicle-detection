'''Utilities
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2


def read_image(img_path):
    if img_path.endswith('.png'):
        img = plt.imread(img_path)*255
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = plt.imread(img_path)

    return img


def color_hist(img, num_bins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=num_bins)
    channel2_hist = np.histogram(img[:, :, 1], bins=num_bins)
    channel3_hist = np.histogram(img[:, :, 2], bins=num_bins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def cal_windows_with_overlap(window_len, dist, overlap):
    '''Calculate the number of windows with overlaps.

    Parameters
    ----------
    window_len
        Length of window assuming square window
    dist
        Total distance
    overlap
        The overlap length between current and previous windows

    Returns
    -------
    num
        Number of windows
    '''
    num = (dist - overlap) // (window_len - overlap)

    return num


def cal_windows_with_step_len(window_len, dist, step_len):
    '''Calculate the number of windows with lenght of step.
    '''
    num = (dist - window_len - step_len) // step_len

    return num


def rescale_image(img, scale):
    if scale != 1:
        imshape = img.shape
        img = cv2.resize(img, (np.int(imshape[1]/scale), 
                               np.int(imshape[0]/scale)))
    return img
