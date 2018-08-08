'''Utilities
'''
import matplotlib.pyplot as plt
import numpy as np


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