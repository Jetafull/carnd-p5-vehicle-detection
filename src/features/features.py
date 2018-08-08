'''Extract histogram of gradients as features.
'''
import cv2
import numpy as np
from skimage.feature import hog


def bin_spatial(img, size=(32, 32)):
    cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features = cv2.resize(img, size).ravel()
    return features


def convert_color_rgb(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        cvt_image = np.copy(image)

    return cvt_image


def hog_features(img, orient, pix_per_cell,
                 cell_per_block, feature_vec=False):
    hog_features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualize=False, feature_vector=feature_vec,
                       block_norm='L2-Hys')

    return hog_features
