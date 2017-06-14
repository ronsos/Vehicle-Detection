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
[image4]: ./output_images/test1_threshold.png
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

| Color Space      | YCrCb  |  
| Orientations     | 11     | 
| Pixels per cell  | 16     | 
| Cells per block  | 2      | 
| HOG channels     |  ALL   |  
| Spatial size     | (16,16) |
| Histogram bins   | 16     |

These settings produced the highest accuracy from the SVC classifier (0.984 - 0.985). 


### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The 4th code block shows the `LinearSVC()` classifier. This is a Support Vector Machine (SVM) classifier, and the SVC stands for Support Vector Classification. `LinearSVC()` works well on large data sets. 

Before being fed into the classifier, the data was scaled using `StandardScaler()`. Then the data was split into a training set and a test set. 80% was randomly put into the training set, and the remaining 20% was put into the test set. The training typically took less than 10 sec to perform, and resulted in an accuracy of about 0.985.


## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

##  Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./P5_project_vid.mp4)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are six frames and their corresponding heatmaps:

![alt text][image5]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

