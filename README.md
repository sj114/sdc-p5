##README
###Vehicle Detection Project

---

**Objectives**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_cars]: ./plots/vehicles.png
[image_noncars]: ./plots/non-vehicles.png
[image_hog]: ./plots/HOG_training_images.png
[image_hog_pipeline]: ./plots/HOG_pipeline_image.png
[image_sliding_w1]: ./output_images/sliding_output_test6.jpg
[image_sliding_w2]: ./output_images/sliding_output_test3.jpg
[image5]: ./plots/test1_heat_label.png
[image6]: ./output_images/output_test1.jpg
[image7]: ./plots/test6_heat_label.png
[image8]: ./output_images/output_test6.jpg
[video1]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 8 through 116 of the file called `feature_extractor.py`).  

The training images comprised of the `vehicle`(GTI and KITTI) and `non-vehicle`(GTI) images provided by Udacity. There were split into training and test data with a 80:20 split using `train_test_split` (line 178-262 of `pipeline.py`). Examples of the vehicular data are as shown:

![alt text][image_cars]
![alt text][image_noncars]

HOG features were extracted by converting each image to a suitable colorspace and applying different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 

Here is an example of a training image using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image_hog]

####2. Explain how you settled on your final choice of HOG parameters.

#####Colorspace
I experimented with RGB, HSV, HLS and YUV colorspaces. I found RGB to do a good job at detecting the cars, but it also threw up a lot of false positives in the surroundings. With HSV, the false positives reduced, but it struggled to identify the white car. Since HLS does a better job at identifying white due to its 'lightness' dimension, it produced the best results out of these experiments.

#####Orientation
After experimenting with various values from 6-9 for the orientation parameter, 8 was chosen for the best tradeoff between vehicle identification vs false positives.

#####Final HOG parameters
* Color space = 'HLS' 
* HOG orientations = 8
* HOG pixels per cell = 8
* HOG cells per block = 2
* HOG channel = ALL
* Spatial binning dimensions = (16, 16)
* Number of histogram bins = 16

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Feature extraction was done using spatial binning (resizing to 16x16), color histogram (bins of 16) and HOG with parameters mentioned above.

These features (number of features: 4344) were fed to a linear support vector machine (from `sklearn.svm`) for classification training using `svc.fit` (lines 309-316 in `pipeline.py`). The test accuracy of the SVC was measured using `svc.score` and came out to be 98.93%. 

####HOG on pipeline
For the images (still and video) being fed to the pipeline, it is inefficient to compute HOG for every sliding window.Instead, the HOG feature matrix is computed for the entire region of interest (bottom half of image) and for every sliding window, the corresponding HOG features are extracted from the larger HOG feature matrix. These features are then flattened and appended to the window-based spatial binning and color histogram features before being fed to the SVM's classifier. (lines 120-176 in `pipeline.py`)

![alt text][image_hog_pipeline]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since objects at the horizon appear to be smaller than objects closer to the camera mounted on the test vehicle, smaller sized windows are used to search the image near the horizon. Similarly, larger sized windows are used below the horizon and the largest sized windows are used near the bottom of the image. 

The region of interest for the sliding window search is restricted to the bottom half of the image, since the top half contains artifacts (sky, trees etc.) above the horizon with no driving region. 

After experimenting with sizes of 48, 64, 96, 128 etc., the final window sizes were chosen as 64, 128, 192. The windows were chosen as multiples of 64 because all images had to be resized to 64 to be fed into the classifier. Since this pipeline does NOT do HOG extraction for every sliding window, and instead computes HOG for the entire region of interest, this meant that HOG vector extraction for any other window size, had to be normalized. This is easier to do for an integer multiple of 64. (lines 120-176 in `pipeline.py`)

The overlap factor is 75% in order to increase the probability of identifying the vehicles, in spite of the additional computational overhead. Experiments were carried out with 50% overlap as well, but this lead to diluted heatmaps making it harder to consistently classify the vehicles. 

Here are examples of the multi-scaled sliding windows superimposed on the camera image:
![alt text][image_sliding_w1]
![alt text][image_sliding_w2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

The prediction of the extracted features into labels (`vehicles` or `non-vehicles`) is done using `svc.predict()`. To reduce false positives, higher weightage is given to those windows whose distance from the deciding hyper-plane is highest, i.e., `svc.decision_function() > threshold`. The threshold is chosen as 0.6 empirically. (line 110 of `pipeline.py`)

In order to further reduce false positives and consolidate overlapping detections, a heatmap was implemented that accrues a weight of 1 at the pixel position for every detection. This is then averaged over 10 successive frames and thresholded at the value of 1.5. Contiguous blobs in the heatmap are identified using `scipy.ndimage.measurements.label()`. The resulting points are then used to plot the bounding boxes. (lines 389-403 of `pipeline.py`)
 
Here are some examples of heatmaps on the provided test images and corresponding labels and output images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4). The algorithm detects some cars in the opposite lanes, but also detects false positives along the edges. 


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Section 2 of the section on *Sliding Window Search* outlines the heatmap and labeling methods used to filter false positives and consolidating overlapping bounding boxes, including details for the video pipeline.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the biggest issues I encountered in this algorithm is the tradeoff between false positives and missing real detections. Further tuning of parameters such as colorspaces or labeling thresholds, or using a better classifier like neural networks might help improve the predictions.

The other issue is the instability in bounding boxes. Although the cars are detected in most frames, the bounding box dimensions fluctuate wildly. A more robust classification will help in accurately identifying the outlines of the bounding boxes. 

####2. Performance

The current algorithm is not real-time. It processes a frame per 1.3s. This is after making optimizations such as computing HOG over the entire frame, and searching subsequent frames in the vicinity of the previously detected boxes (fresh search occurs every 10 frames).

Here is the breakdown for the frame processing:
* 0.33 Seconds to  get HOG
* 0.36 Seconds to  get sliding windows (64)
* 0.16 Seconds to  get hot windows
* 0.17 Seconds to  get sliding windows (128)
* 0.05 Seconds to  get hot windows
* 0.12 Seconds to  get sliding windows (192)
* 0.02 Seconds to  get hot windows
* 0.02 Seconds to  draw/heat windows
= 1.23 seconds for a full frame
 
Clearly, getting HOG and sliding windows takes the maximum amount of time. Reducing feature length and number of windows would boost performance, but this would need to be done carefully in order to ensure that it doesn't affect detections.

####3. Summary
Overall, this algorithm detects vehicles in a video stream, with some false positives and instabilities in dimensions of bounding boxes.
