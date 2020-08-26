# Prostate Cancer Grade Assessment
Using GANs to detect cancerous regions from prostate cancer biopsy images.

## Introduction
With more than 1 million new diagnoses reported every year, prostate cancer (PCa) is the second most common cancer among males worldwide that results in more than 350,000 deaths annually. The key to decreasing mortality is developing more precise diagnostics. Diagnosis of PCa is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the Gleason grading system.This repository is an attempt to deploy Pix2Pix for highlighting cancerous region on images of prostate tissue samples, and estimate severity of the disease using OpenCV.

## Approach
The outcome of the model is to input a biopsy image and the model will deliver a newly generated mask image that can capture the cancerous regions. This repository approaches the problem as an image-to-image translation problem. The [Pix2pix](https://arxiv.org/abs/1611.07004) model was trained on two sets of images: the original biopsy image and the mask image that highlight the cancerous regions. 
![Screen Shot 2020-08-25 at 18 11 55](https://user-images.githubusercontent.com/64785877/91252853-99cae180-e788-11ea-8956-e9a0a3c72da2.jpg)


## Dataset
The dataset used in this repository is from a Kaggle competition: Prostate cANcer graDe Assessment (PANDA) Challenge. 

[Dataset Link](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview)

## Data cleaning/preprocessing
The dataset posed some problems that needed to be preprocess before putting in the model for training:

The mask images' pixel range is not from 0-255 like normal RGB image but range from 0-5 as label for each stage of the cancer.
This happens for the following two reasons :

  - The label information is stored in the red (R) channel, the other channels are set to zero and can be ignored.

  - The masks are not image data like the WSIs.They are just matrices with values based on the data provider information provided above, instead of containing a range of values from 0 to 255, they only go up to a maximum of 6, representing the different class labels. Therefor when you try to visualize the mask, it will appear very dark as every value is close to 0. 
  
Solutions: Applying the color map fixes the problem by assigning each label between 0 and 6 a distinct color and save the new images.

## Training
We will follow the same training procedure outlined in [Pix2Pix](https://github.com/phillipi/pix2pix) example with the preprocessed dataset.

## Result 
![Screen Shot 2020-08-17 at 20 34 40](https://user-images.githubusercontent.com/64785877/91165737-62622380-e6fb-11ea-8315-97b6c8ec5009.jpg)

The image on the left is a biopsy image, the image in the middle is a referenced mask image provided by the data provider, the right image is the model-generated image.

## File Structure 
### [Change-to-png.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/change_to_png.py)
This file is responsible for changing all the images from tiff format into png. This file also contains solution to the mask images problem. 

### [Remove_extra_imgs.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/remove_extra_imgs.py)
This file is reponsible for removing all extra biopsy images that doesn't have a corresponding mask image.

### [Clean_up_white_img.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/clean_up_white_img.py)
This file is reponsible for filtering out all-white or all-black images.

### [Helper_functions.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/helper_functions.py)
This file is responsible for defining all miscellaneous helper functions.

### [Predict.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/predict.py)
This file is responsible for predicting the labels for the input data.

### [Preprocess.py](https://github.com/AiNguyen237/Prostate-cancer-grade-assessment/blob/master/preprocess.py)
This file is responsible for preprocessing the input data.

