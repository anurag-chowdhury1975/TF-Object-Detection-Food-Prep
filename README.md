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

Execute **Generate_Images.py** to generate images from a live video recording or a prerecorded video. Commented out code can be used to switch from prerecorded video to live video to generate the images. 
The code will save every 10th frame in the video as an .jpg image in a folder called "Images" under your project root folder. You can change this setting to capture more/fewer frames as you see fit. You can also hit the "p" key to manually capture images.

### Part 2. Annotate the Images

Each of the images extracted from the video frames in Part 1 will need to be annotated. I used the opensource annotation tool **labelImg** to create object bounding boxes and labels for each object in the images. 
LabelImg supports both PASCAL VOC and YOLO annotation formats. I used PASCAL VOC format which creates an XML file of the image annotations and labels. 
These XML annotation files are stored in the "/Annotations" folder.
Execute **XML_to_CSV.py** to combine the XML annotation files for each image into one csv file (labels.cs). This CSV file is saved in the same "/Annotations" folder as the XMl files.

**LabelImg GitHub:** (https://github.com/HumanSignal/labelImg ).
How to install and use labelImg - https://www.youtube.com/watch?v=fjynQ9P2C08

**NOTE:** labelImg package has an issue with float values which will need to be corrected in the source code. Follow the instructions here to fix the error so you can go ahead and label your images - https://www.youtube.com/watch?v=5jHPuwo8z1o 

### Part 3. Augment the Images so you have more images to train and test your model on

Execute **Augment_Images.py** to augment the original images (generated from part 1) and also update the corresponding annotations "labels.csv" file (generated from Part 2). Each of the original images is augmented in to 6 images (5 for Train and 1 for Test at random). You can change how many augmented images you want as you see fit.

The augmented images are split into train and test batches and saved to "/Images_Aug/Train" and "/Images_Aug/Test" folders respectively. The corresponding annotation csv files (labels_aug_train.csv, labels_aug_test.csv) are saved in the "/Annotations" folder.

**Reference:** https://alex-vaith.medium.com/save-precious-time-with-image-augmentation-in-object-detection-tasks-2f9111abb851

**NOTE:** the imgaug version uses an older version of Numpy (< 1.16) and in the new version of Numpy the np.sctypes() function has been deprecated. Due to this you will have to located the imgaug.py file and update lines 45-47 –
Change:
NP_FLOAT_TYPES = set(np.sctypes["float"])
NP_INT_TYPES = set(np.sctypes["int"])
NP_UINT_TYPES = set(np.sctypes["uint"])
To:
NP_FLOAT_TYPES = set([np.float32, np.float64, np.float16, np.longdouble])
NP_INT_TYPES = set([np.int8, np.int16, np.int32, np.int64, np.int_, np.intc, np.intp, np.integer])
NP_UINT_TYPES = set([np.uint8, np.uint16, np.uint32, np.uint64, np.unsignedinteger])

### Part 4. Fine tuning the object detection model 

Fine tuning the pre-trained EfficientDet object detection model was done on Google Colab as I did not have GPU/TPU resources on my local machine.

You will first need to upload the following files (and folder) to your Google drive -
- labels_aug_train.csv
- labels_aug_test.csv
- Images_Aug/*.*
- generate_tfrecord.py
- labelmap.pbtxt

Execute the sections in the Python notebook to create the TF record, download the TensorFlow pre-trained object detection models in your Google Colab environment and fine tune the model for your particular usecase - **/Model_Training/Tensorflow_2_Object_Detection_Model_Training_2.ipynb**

You will first need to upload the following files (and folder) to your Google drive -
- labels_aug_train.csv
- labels_aug_test.csv
- Images_Aug/*.*
- generate_tfrecord.py
- labelmap.pbtxt

**NOTE:** You will need to update the function "class_text_to_int" in generate_tfrecord.py and the labelmap.pbtxt files to match the class/labels specific to your usecase.

**Tensorflow models:** https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

EfficientDet architecture and code - https://github.com/google/automl/tree/master/efficientdet (refer to section 11 in the readme.txt to see how to reduce GPU memory usage)

### Part 5. Test the custom fine tuned object detection model 
Download the saved model from Colab (or wherever you choose to train) to your local machine project root folder /Models/.

Execute **Test_Custom_ODM_Vid.py** to test the fine tuned model on new videos/images (that were not used in the training but have the same set of objects in them).



