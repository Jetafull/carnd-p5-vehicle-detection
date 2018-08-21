# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ../images/output_images/hog_visualization.png
[image2]: ../images/output_images/hog_rgb.png
[image3]: ../images/output_images/hog_yuv.png
[image4]: ../images/output_images/slide_window_64.png
[image5]: ../images/output_images/heatmap1.png
[video1]: ../videos/submission_videos/project_video.mp4

### Structure of The Project

The project includes the following files:

1. `notebooks/prototype.ipynb`: The code for developing a solution for object detection.
2. `notebooks/pipeline.ipynb`: An implementation of the pipeline to detect cars in videos.
3. `reports/writeup.md`: The report for the project.
4. `images/output_images`: Images used in the writeup file.
5. `videos/submission_video/project_video.mp4`: The video after processed by the car detector.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `HOG Classifier` and `Color Space` parts in `notebooks/pipeline.ipynb`. I explored different combinations of parameters for HOG. This is an example with `orientation=9`, `pixels_per_cell=16` and `cells_per_block=2`.

![hog_visualization][image1]

I also compare different color spaces before conducting HOG. Here is an example with the RGB color space:

![hog_rgb][image2]

Here is an example with YUV color space:

![hog_yuv][image3]

I found the YUV color space captures different features from the image. For example, in the HOG visualization above the Y channel capture the overall brightness change. The U and V channel capture the gradient change because of color difference. The V channel captures the gradient change around the back light and gives the gradients in that cell higher magnitude.

Each channel in RGB color space gives more similar HOG visualization than the YUV color space. I chose the YUV color space because it gives a more accurate image segmentation in this task.

#### 2. Explain how you settled on your final choice of HOG parameters.

I chose the HOG parameters based on two factors:

1. The dimension of feature vector.
2. The accuracy of classifier.

I found a higher number of orientation (`orientation`) helps to improves the classifier performance. However, increasing number of orientation can increase the dimensions of the feature vector rapidly. I found 9 to 11 are a range for better result. In this case, I chose 10 as the number of orientation.

I also change the `pixels_per_cell` from 8 to 16 as I found the classifier performs almost the same while the computation time is shortened after increasing this number.

My final choice of HOG parameters are:

1. color space: YUV
2. number of orientations: 10
3. pixels_per_cell: 16
4. cells_per_block: 2

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in `Feature Classify` part in `notebooks/pipeline.ipynb`.

I trained a linear SVM using the features below:

1. Spatial features with spatial size `(16, 16)`
2. Histograms features with 16 bins
3. HOG features

The three types of features are combined after normalization. The dimension of the feature vector is 1896.

To train the model, I first manually select the vehicle images from the GTI datasets and combine them with the KITTI dataset. My final dataset has 6920 vehicle images and 8972 non-vehicle images. I randomly split the dataset by 80/20 ratio for training and validation.

To prevent overfitting, I set the `C=0.01` in the linear SVC classifier. The accuracy on the validation set is above 99%.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this part is contained in the `Sliding windows` and `Multiple detections and false positive` in the `notebooks/pipeline.ipynb`.

On each frame of the video, I applied multiple sizes of windows and record the positive detections from each image. After summing up the positive detections, I applied a threshold on the final heatmap to reduce the number of false positives.

Here is an example of apply the window with size `(64, 64)` and `overlap=0.75`.

![false_positive_individual][image4]

I used the `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I draw a bounding boxes to cover the area of each blob detected.

Here is an example after summing up the heatmap from different sizes of windows and drawing the bounding boxes:

![bounding box][image5]

The left image is the result bounding boxes after applying thresholds and labeling. The right image is the corresponding heatmap.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is slow to process the image. I have used the following ways to speed up the detection:

1. Increase the number of pixels per cell to reduce the dimension of feature vector.
2. Further cropping the image on the x-axis direction to reduce the windows to be processed.
3. Reduce the spatial size for spatial features.

These methods speed up the detection processing but my current implementation is still not fast enough for near real-time processing on the video. To improve further the processing speed, we can try:

1. Multi-threading on different scales during processing.
2. Perform HOG on once and then extract the HOG features by subsampling from the image.

There are still some false positives in detection. One reason is the dataset is still relatively small. We can improve this by augmenting the dataset using different brightness, perspectives and viewing angles.

One issue in the video processing is when two cars are too close, the bounding box will be merged and two cars are identified as a single car. One reason is that we are tracking the objects by its relative positive in the image. This can be misleading since when a different car comes to the same position, it will keep using the records of the previous car. One possible solution is when a new object appears we reset the heatmap records in the surrounding area.