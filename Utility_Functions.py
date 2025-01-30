import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import xml.etree.ElementTree as ET

from object_detection.utils import ops as utils_ops
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree

def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(" ")[-1].replace("\"", " ").strip()
            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects Elektrische Bauteile batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def select_best_bbs(output_dict, min_score_threshold, max_num_boxes, max_iou_threshold):
  # Assume 'detection_boxes', 'detection_scores', and 'detection_classes' are the outputs from the model inference
  # Example format of model inference output
  # detection_boxes: [num_detections, 4] tensor of bounding boxes
  # detection_scores: [num_detections] tensor of detection scores
  # detection_classes: [num_detections] tensor of detection class indices

  detection_boxes = output_dict['detection_boxes']
  detection_classes = output_dict['detection_classes']
  detection_scores = output_dict['detection_scores']

  # Convert the outputs to numpy arrays (if they are tensors)
  # detection_boxes = detection_boxes.numpy()
  # detection_scores = detection_scores.numpy()
  # detection_classes = detection_classes.numpy()
  # Filter out the detections with a score lower than the threshold
  valid_detections_mask = detection_scores >= min_score_threshold
  filtered_boxes = detection_boxes[valid_detections_mask]
  filtered_scores = detection_scores[valid_detections_mask]
  filtered_classes = detection_classes[valid_detections_mask]

  # Apply non-maximum suppression
  selected_indices = tf.image.non_max_suppression(
      boxes=filtered_boxes,
      scores=filtered_scores,
      max_output_size=max_num_boxes,  # Maximum number of boxes to be selected by NMS
      iou_threshold=max_iou_threshold,  # Intersection-over-union threshold
      score_threshold=min_score_threshold
  )

  # Select the boxes, scores, and classes using the indices from NMS
  selected_boxes = tf.gather(filtered_boxes, selected_indices).numpy()
  selected_scores = tf.gather(filtered_scores, selected_indices).numpy()
  selected_classes = tf.gather(filtered_classes, selected_indices).numpy()

  return selected_boxes, selected_scores, selected_classes

def get_image_with_actual_lbl_box(img: np.ndarray, img_name: str, img_boxs: pd.DataFrame) -> np.ndarray:
    """
    This function will:
    1. get all bounding boxes that relate to the image for the DataFrame
    2. concatenate them with imgaug package to a BoundingBoxesOnImage object
    3. draw the Bounding boxes onto the image
    :param img: image as np array
    :param img_name: filename to locate the bounding boxes in the df
    :param df: DataFrame that holds the information about all bounding boxes
    :return: image with bounding boxes drawn onto it
    """
    # create bounding box with imgaug

    bbs = list()
    for _, row in img_boxs.iterrows():
        x1 = row.xmin
        y1 = row.ymin
        x2 = row.xmax
        y2 = row.ymax
        label = row['class']
        # bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        img = bb.draw_box_on_image(img, color=(255, 0, 0), alpha=1.0, size=1, copy=True, raise_if_out_of_image=True, thickness=None)
        img = bb.draw_label_on_image(img, color=(255, 0, 0), color_text=None, color_bg=None, alpha=1.0, size=1, size_text=12, height=35, copy=True, raise_if_out_of_image=False)

    # convert single bounding boxes to BOundingBoxOnImage instance to draw it on the picture
    # bbs = BoundingBoxesOnImage(bbs, img.shape[:-1])

    # draw image
    # img = bbs.draw_on_image(img, color=(0, 255, 0), alpha=0.5, raise_if_out_of_image=True)
    return img

def get_image_with_pred_lbl_box(img, detection_boxes, detection_scores, detection_classes, label_map):
    bbs = list()
    for i in range(0,len(detection_scores)):  
        label = label_map.get(detection_classes[i]) + "\n(" + str(round(detection_scores[i]*100,2)) + "%)"
        y1 = img.shape[0] * detection_boxes[i][0]
        x1 = img.shape[1] * detection_boxes[i][1]
        y2 = img.shape[0] * detection_boxes[i][2]
        x2 = img.shape[1] * detection_boxes[i][3]
        # bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        img = bb.draw_box_on_image(img, color=(255, 0, 0), alpha=1.0, size=1, copy=True, raise_if_out_of_image=True, thickness=None)
        img = bb.draw_label_on_image(img, color=(255, 0, 0), color_text=None, color_bg=None, alpha=1.0, size=1, size_text=12, height=35, copy=True, raise_if_out_of_image=False)
    # convert single bounding boxes to BOundingBoxOnImage instance to draw it on the picture
    # bbs = BoundingBoxesOnImage(bbs, img.shape[:-1])

    # draw image
    # img = bbs.draw_on_image(img, color=(255, 0, 0), alpha=0.5, raise_if_out_of_image=True)
    # img = bbs.draw_box_on_image(img, color=(0, 255, 0), alpha=1.0, size=1, copy=True, raise_if_out_of_image=True, thickness=None)
    return img

# Resize an image to a square size (to train model) keeping the same aspect ratio and adding padding as required
def resize_with_aspect_ratio(image, size):
    h, w = image.shape[:2]

    # Calculate the scaling factor to maintain aspect ratio
    scale = size / max(h, w)

    # Resize the image with the scaling factor
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new square image with the desired size and a white background (or any color)
    padded_image = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Calculate padding
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2

    # Place the resized image onto the center of the padded image
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image

    return padded_image, (new_h, new_w), (pad_h, pad_w)

def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not subelem.tail or not subelem.tail.strip():
            subelem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# Create a PASCAL VOC XML file for detected objects in an image
def write_xml(folder, filename, bbox_list, img_width, img_height):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = './' + folder + '/' + filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(img_width)
    SubElement(size, 'height').text = str(img_height)
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for obj in bbox_list.items():
        bbox = obj[1]
        
        obj_class_name = "class1"
        obj_xmin = bbox[0][0]
        obj_ymin = bbox[0][1]
        obj_xmax = bbox[1][0]
        obj_ymax = bbox[1][1]
            
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = obj_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(obj_xmin)
        SubElement(bbox, 'ymin').text = str(obj_ymin)
        SubElement(bbox, 'xmax').text = str(obj_xmax)
        SubElement(bbox, 'ymax').text = str(obj_ymax)

    indent(root)
    tree = ElementTree(root)

    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)