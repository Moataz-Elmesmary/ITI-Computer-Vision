### Day Night Image Classifier

###### Problem Statement
Given an outdoor image, classify whether it was captured during day or night, with good level of confidence and accuracy.

###### Dataset
The dataset used for the project is a smaller version of the huge [AMOS dataset](http://cs.uky.edu/~jacobs/datasets/amos/) *(Archive of Many Outdoor Scenes)*.

###### Aim
To build a classifier that can accurately label these images as day or night, and that relies on finding distinguishing features between the two types of images!

###### Training & Testing Data
The 400 total images are separated into training and testing datasets.

**60%** of these images are training images, used it to create a classifier.
**40%** are test images, used to test the accuracy of the classifier.
These are some variables to keep track of where our image data is stored:

```
image_dir_training: the directory where our training image data is stored
image_dir_test: the directory where our test image data is stored
IMAGE_LIST: list of training image-label pairs
STANDARDIZED_LIST: list of preprocessed image-label pairs
```