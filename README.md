# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps, normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/test1_bboxes.png
[image4]: ./output_images/test6_bboxes.png
[image5]: ./output_images/test1_threshold.png
[image6]: ./output_images/test6_threshold.png
[image7]: ./output_images/test1_heat.png
[image8]: ./output_images/test6_heat.png
[video1]: ./P5_project_vid.mp4

Each of the points in the project [rubric](https://review.udacity.com/#!/rubrics/513/view) will be addressed in the description below. 

--

##  Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This README serves this purpose. 

## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The first two code cells contain functions that were provided in the lectures. The third cell contains the code for loading the image data and doing some example HOG feature extractions and visualizations. 
 
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![carnotcar][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space of the `vehicle` class example above. The HOG parameters used for feature extraction were `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![car_hog][image2]


### 2. Explain how you settled on your final choice of HOG parameters.

Through some internet research I was able to determine that using `YUV` or `YCrCb` were most likely to be successful. I initially tried `YUV` but saw an error in channel 2 that I never fully resolved. `YCrCb` seemed to be working well (much better than `RGB`), so I began working with that color space and started tweaking the other parameters. The information provided by a reviewer and shared by a student [here](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903) was helpful in determining the settings for the feature extraction. 

I selected the following settings through trial and error:

| Parameter | Setting |
|:-------------:|:------:|
| Color Space      | YCrCb  |  
| Orientations     | 11     | 
| Pixels per cell  | 16     | 
| Cells per block  | 2      | 
| HOG channels     |  ALL   |  
| Spatial size     | (16,16) |
| Histogram bins   | 16     |

These settings produced the highest accuracy from the SVC classifier (0.987). 


### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The 4th code block shows the `LinearSVC()` classifier. This is a Support Vector Machine (SVM) classifier, and the SVC stands for Support Vector Classification. `LinearSVC()` works well on large data sets. 

Before being fed into the classifier, the data was scaled using `StandardScaler()`. Then the data was split into a training set and a test set. 80% was randomly put into the training set, and the remaining 20% was put into the test set. The training typically took less than 10 sec to perform, and resulted in an accuracy of about 0.987.


## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The 5th cell shows the sliding window search implementation. Window size of (96, 96) was selected along with an overlap of 0.75. These settings were chosen by trial and error involving iterating and improving the algorithm performance based on the video results. 

These images show the returns from the sliding windows search. They are from the provided test_images set, `test1.png` and `test6.png` respectively.  

![test1][image3]
![test6][image4]


### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

A threshold is applied to the result from the sliding windows search. The thresholding is set so that it only returns areas of overlapping boxes. These images are the same two test images as above, but now with thresholding applied so that a single box is returned for each identified vehicle. 

Note that, as these images show, this thresholded result does not always perfectly encompass the vehicle. It can be tricky to come up with a single threshold value for the entire video. It may be that a dynamic thresholding approach would improve performance.  

Threshold as applied to `test1.png` and `test6.png` respectively.  

![test1][image5]
![test6][image6]

---

##  Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./P5_project_vid.mp4)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The set of bounding boxes from the sliding windows search is used to generate a heat map. Basically, for any pixel inside a bounding box, a value of 1 is added to the heat map for that pixel (starting from zero for all pixels). Each of the bounding boxes is considered when applying the heat map. Thus, any area of the frame that has multiple overlapping bounding boxes will generate a heat map value of greater than 1. 

The heat map results for test images `test1.png` and `test6.png` are as follows: 

![test1][image7]
![test6][image8]

When analyzing the video, a simple frame by frame threshold was not sufficient for identifying the vehicles while keeping out false positives. Way too many false positives would show up with a threshold that was low enough to capture the vehicles. 

The approach that I selected was to use a `collections.deque` array for smoothing. This can be seen in the 6th cell, labeled "Pipeline". The size of the array was set to 10. For thresholding, the sum of the 10 heat maps was determined. If a particular pixel exceeded a threshold of 6, then it was applied to the final heat map that was used for generating the bounding boxes for that single frame of the video. The bounding boxes are found using the `scipy.ndimage.measurements.label()` function, overlaid on the image frame from the video, then stitched together to make the modified video showing the bounding boxes. 


---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The video result shows zero false positives, which is very good. However, it occasionally will drop a frame or two on the vehicle identification. This is the correct side of the ledger to err on, but the algorithm could be improved so that it does not lose the vehicle at all. The current thresholding is set as low as possible, i.e. it is the lowest value (6) that eliminated all false positives. One approach that might help is to increase the size of the collections.deque array so that the averaging is done over a large number of frames. This would allow a bit more fine control of the thresholding. However, it would slow down the video processing, probably by a fairly significant amount. 

A second potential improvement is tighter bounding boxes around the vehicles. Some of the frames show boxes that are too small, others too large. A tighter resolution on the sliding windows search or more overlap could help with this. The threshold would need to be readjusted to fine tune this approach. (Full disclosure: I attempted this, but was unable to achieve a satifactory result). 
