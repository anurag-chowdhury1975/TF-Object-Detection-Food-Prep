import os
import cv2
import random
import pandas as pd
import numpy as np

import Utility_Functions as uf
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


# specify augmentations that will be executed on each image randomly
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.2), # vertical flip
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.05))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.95, 1.05)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.95, 1.05), per_channel=0.25),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
         scale={"x": (1, 1.5), "y": (1, 1.5)},
         translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
         rotate=(-3, 3),
    )],
    random_order=True) # apply augmenters in random order# apply augmenters in random order


def aug_image(filename: str, df: pd.DataFrame, folder: str, augmentations: int) -> tuple[list, list]:
    """
    This function will:
     1. load the image based on the filename from the given folder
     2. load all given bounding boxes to that image from the given DataFrame
     3. apply augmentations specified by the seq variable above
     4. output images and bounding_boxes
    :param filename: str object that defines the image to be augmented
    :param df: DataFrame that stores all given bounding box information to each image
    :param folder: defines where to find the image
    :param augmentations: defines the number of augmentations to be done
    :return: list of augmented images, list of bouding_boxes for each augmented image
    """
    # load image
    img = cv2.imread(os.path.join(folder, filename))
    # create empty list for bounding_boxes
    bbs = list()
    # iterate over DataFrame to get each bounding box for that image
    for _, row in df[df.filename == filename].iterrows():
        x1 = row.xmin
        y1 = row.ymin
        x2 = row.xmax
        y2 = row.ymax
        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=row['class']))
    # concatenate all bounding boxes fro that image
    bbs = BoundingBoxesOnImage(bbs, shape=img.shape[:-1])

    # replicate the image augmentations times
    images = [img for _ in range(augmentations)]
    # replicate the bounding boxes augmentations times
    bbss = [bbs for _ in range(augmentations)]

    # augment images with bounding_boxes
    image_aug, bbs_aug = seq(images=images, bounding_boxes=bbss)

    return image_aug, bbs_aug


def save_augmentations(images: list, bbs: list, df: pd.DataFrame, filename: str, folder: str, resize: bool = False,
                       shape: tuple[int, int] = (None, None)) -> pd.DataFrame:
    """
    This function will:
    1. store each augmented image in a new folder
    2. append the bounding_boxes from the augmented_images to the given DataFrame
    :param images: list of augmented images
    :param bbs: list of concatanted bounding boxes that relate to an augmentated image
    :param df: DataFrame that will store the information about the new bounding boxes from the augmented images
    :param filename: original filename of the original image
    :param folder: str object that defines the path to the output folder for the augmentated images
    :param resize: defines if the image should be resized or not after the augmentation
    :param shape: if the image will be reshaped, it will be reshaped into this shape
    :return: DataFrame
    """

    # iterate over the images
    for [i, img_a], bb_a in zip(enumerate(images), bbs):
        # define new name
        aug_img_name = f'{filename[:-4]}_{i}.jpg'
        # check if image should be resized
        org_shape = (None, None)
        if resize:
            org_shape = img_a.shape[:-1]
            # if resizing to a square image then resize keeping the aspect ratio the same as original image with padding
            if shape[0] == shape[1]:
                img_a, unpadded_size, padding = uf.resize_with_aspect_ratio(img_a, shape[0])
            else:
                img_a = cv2.resize(img_a, shape, interpolation=cv2.INTER_AREA)

        # clean bb_a --> use only bounding boxes that are still in the frame (cropping can lead to bounding boxes being
        # removed from the images)
        bb_a = bb_a.remove_out_of_image().clip_out_of_image()

        # iterate over the bounding boxes
        at_least_one_box = False
        for bbs in bb_a:
            if resize:
                # if resizing to a square image then then use the unpadded resized image size
                if shape[0] == shape[1]:
                    bbs = bbs.project(org_shape, unpadded_size)
                    bbs.x1 += padding[1]
                    bbs.y1 += padding[0]
                    bbs.x2 += padding[1]
                    bbs.y2 += padding[0]
                else:
                    bbs = bbs.project(org_shape, shape)
            arr = bbs.compute_out_of_image_fraction(img_a)
            if arr < 0.8:
                at_least_one_box = True
                x1 = int(bbs.x1)
                y1 = int(bbs.y1)
                x2 = int(bbs.x2)
                y2 = int(bbs.y2)
                c = bbs.label
                # append extracted data to the DataFrame
                height, width = img_a.shape[:-1]
                df = pd.concat([df, pd.DataFrame(data=[aug_img_name, width, height, c, x1, y1, x2, y2],
                                            index=df.columns.tolist()).T])
        if at_least_one_box:
            # save image at specified folder
            cv2.imwrite(os.path.join(folder, aug_img_name), img_a)

    return df


if __name__ == '__main__':
    # specify folder
    folder = 'Images'
    # define number of augmentations per image
    augmentations = 5
    # specify if the image should be resized
    resize = True
    # define shape (should be equal to requested shape of the object detection model
    new_shape = (512, 512)
    # define input folder
    input_folder = os.path.join('.', folder)
    # define and create output_folder
    output_folder_train = os.path.join('.', f'{folder}_Aug/Train')
    output_folder_test = os.path.join('.', f'{folder}_Aug/Test')
    if not os.path.isdir(output_folder_train):
        os.makedirs(output_folder_train)
    if not os.path.isdir(output_folder_test):
        os.makedirs(output_folder_test)

    # 1. get a list of all images in the folder
    img_list = [img for img in os.listdir(input_folder) if img.endswith('.jpg')]

    # 2. load DataFrame with annotations
    df = pd.read_csv(os.path.join('.', 'Annotations/labels.csv'))
    # create a new pandas table for the augmented images' bounding boxes
    aug_df_train = pd.DataFrame(columns=df.columns.tolist())
    aug_df_test = pd.DataFrame(columns=df.columns.tolist())

    # 2. iterate over the images and augmentate them
    for filename in img_list:
        # augment image
        aug_images, aug_bbs = aug_image(filename, df, input_folder, augmentations)

        # Choose at random 1 image out of the total augmentations for Test and the remaining for Training
        # In this case total augmentations per image was set to 5 (see above) and therefore 4 will be used to Training and 1 for Testing the model
        test_index = random.randint(0, augmentations-1)
        aug_images_test = [aug_images[test_index]]
        aug_images_train = aug_images[0:test_index] + aug_images[test_index+1: augmentations]
        aug_bbs_test = [aug_bbs[test_index]]
        aug_bbs_train = aug_bbs[0:test_index] + aug_bbs[test_index+1: augmentations]

        # store augmentations in new DataFrame and save image
        aug_df_train = save_augmentations(aug_images_train, aug_bbs_train, aug_df_train, filename, output_folder_train, resize, new_shape)
        aug_df_test = save_augmentations(aug_images_test, aug_bbs_test, aug_df_test, filename, output_folder_test, resize, new_shape)

    # save new DataFrame
    aug_df_train.to_csv(os.path.join('.', 'Annotations/labels_aug_train.csv'), index=False)
    aug_df_test.to_csv(os.path.join('.', 'Annotations/labels_aug_test.csv'), index=False)