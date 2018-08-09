''' Extract features by subsampling on the images.

Extract base features once and then calculate the final features
by subsampling.
'''
import numpy as np
import cv2
from features import extract_hog_features, bin_spatial, color_hist
from utils import convert_color_rgb


def extract_subsample_features(img, window, cspace, cells_per_step, ystart,
                               ystop, scale, classifier, X_scaler, orient,
                               pix_per_cell, cell_per_block, spatial_size,
                               hist_bins):
    '''Extract features by subsampling.

    Parameters
    ----------
    img
        Image array.
    window
        The length of window border (squared).
    cspace
        The color space.
    cells_per_step
        Number of cells to move per step.
    ystart
        Start position for subsampling.
    ystop
        Stop position for subsampling.
    scale
        Scale of the image.
    classifier
        Classifier to predict on each window
    orient
        Number of orientations in each cell
    pix_per_cell
        Pixels in each cell
    cell_per_block
        Number of cells in each block
    spatial_size
        Spatial size
    hist_bins
        Bins for color space histograms

    Returns
    -------
    '''
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color_rgb(img_tosearch, cspace=cspace)

    # rescale image
    ctrans_tosearch = rescale_image(ctrans_tosearch, scale)

    # separate images to channels
    ch1 = img[:, :, 0]
    ch2 = img[:, :, 1]
    ch3 = img[:, :, 2]

    # Compute individual channel HOG features for the entire image
    hog1 = extract_hog_features(ch1, orient, pix_per_cell, cell_per_block)
    hog2 = extract_hog_features(ch2, orient, pix_per_cell, cell_per_block)
    hog3 = extract_hog_features(ch3, orient, pix_per_cell, cell_per_block)

    nxsteps, nysteps, nblocks_per_window = \
        subsample_hog(ch1, window, orient, pix_per_cell,
                      cell_per_block, cells_per_step)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop+window, xleft:xleft+window],
                (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)))
            test_features = test_features.reshape(1, -1)
            test_prediction = classifier.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,
                              (xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw, ytop_draw+win_draw+ystart),
                              (0, 0, 255), 6)
    return draw_img


def subsample_hog(ch1, window, pix_per_cell, cell_per_block, cells_per_step):

    # Define blocks and steps
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    return nxsteps, nysteps, nblocks_per_window


