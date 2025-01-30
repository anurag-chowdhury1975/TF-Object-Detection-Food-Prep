# Ingredient Detection in Food Preparation

This project is a proof of concept demo of how CV Object Detection models can be used to detect the ingredients in food preperation. A solution like this can be adopted by restaurants and cloud kitchens to monitor food preperation and flag orders that may have been prepared incorrectly (i.e. if a required ingredient was missed or an ingredient not part of the recipe was used).

**NOTE:**

I learnt from following repos [BenGreenfield825](https://github.com/BenGreenfield825/Tensorflow-Object-Detection-with-Tensorflow-2.0), [Gilbert Tanner](https://github.com/TannerGilbert/Tensorflow-Object-Detection-with-Tensorflow-2.0) and reused much of their code to the ingredient detection specific problem I was trying to solve.

This project leverages the pretrained EfficientDet (512X512) Tensorflow Object Detection model that was fine tuned on the Burger preperation use case. 

Here is a demo of the ingredient detection using the fine tuned object detection model:

https://github.com/user-attachments/assets/85e2e56c-840c-4f36-a1ee-461c86418e92

## Authors

[Anurag Chowdhury](https://www.linkedin.com/in/anurag-chowdhury-8752377/)

## Setup Details

The [Setup](requirements.txt) file contains all the necessary python packages and other details if you need to replicate what I did.  

## How to work with this repo

### Part 1. Preparing Images to finetune the Tensflow object detection model 

Execute **Generate_Images.py** to generate images from a live video recording or a prerecorded video.
The code will save every 10th frame in the video as an .jpg image in a folder called "Images" under your project root folder. You can change this setting to capture more/fewer frames as you see fit. You can also hit the "p" key to manually capture images.

### Part 2. Augment the Images so you have more images to train and test your model on

Execute **Augment_Images.py** to augment the original images. Each of the original images is augmented in to 6 images (5 for Train and 1 for Test at random). You can change how many augmented images you want as you see fit.



