def train():
    CAR_IMAGES_PATH = ROOT_PATH/'data/vehicles/all'
    NOTCAR_IMAGES_PATH = ROOT_PATH/'data/non-vehicles/GTI'
    NOTCAR_IMAGES2_PATH = ROOT_PATH/'data/non-vehicles/Extras'

    cspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)
    hist_bins = 16
    spatial_feat = False
    hist_feat = False
    hog_feat = True

    car_features = []
    print('car features...')
    car_files = list(CAR_IMAGES_PATH.glob('*png'))
    num_files = len(car_files)
    for file in np.array(car_files):
        img = read_image(file.as_posix())
        features = extract_features(img, cspace=cspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
        car_features.append(features)

    print('notcar features...')
    notcar_features = []
    noncar_files = list(NOTCAR_IMAGES_PATH.glob('*png')) + \
        list(NOTCAR_IMAGES2_PATH.glob('*png'))
    num_files = len(noncar_files)
    for file in np.array(noncar_files)[idx]:
        img = read_image(file.as_posix())
        features = extract_features(img, cspace=cspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features.append(features)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    
    return X, y


if __name__ == '__main__':
    train()
