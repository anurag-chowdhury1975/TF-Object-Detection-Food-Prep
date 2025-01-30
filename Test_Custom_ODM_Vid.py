import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import Utility_Functions as uf

from subprocess import PIPE, run
from object_detection.utils import ops as utils_ops
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

tf.keras.backend.clear_session()
model = tf.saved_model.load('./Models/saved_model')

BURGER_PREP_SERIES = sys.argv[1]
VIDEO_STATUS = sys.argv[2]
label_map = uf.read_label_map('./labelmap.pbtxt')

# Check the 2nd command line argument
if VIDEO_STATUS is not None and VIDEO_STATUS == "p": # if "p" then predict on a pre-recorded video
    cap = cv2.VideoCapture("Videos/burger_prep_demo_"+BURGER_PREP_SERIES+".avi")
else: # Default to predict on a live video capture
    device_id = 0
    webcam_name = "HD Pro Webcam C920" # External webcam name
    # webcam_name = "FaceTime HD Camera" # Built in mac camera name

    command = ['ffmpeg','-f', 'avfoundation','-list_devices','true','-i','""']
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    for item in result.stderr.splitlines():
        print(item)
        if webcam_name in item:
            device_id = int(item.split("[")[2].split(']')[0])
            break

    print(device_id)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Width = {}, Height = {}".format(width, height))
# if "r" then save record the live video capture
if VIDEO_STATUS is not None and VIDEO_STATUS == "r": # If instructed to record video
    video_rec = cv2.VideoWriter("Videos/burger_prep_demo_"+BURGER_PREP_SERIES+".avi", cv2.VideoWriter_fourcc('X','V','I','D'), 
                                fps=int(cap.get(cv2.CAP_PROP_FPS)), frameSize=(width, height))
    video_rec_bbs = cv2.VideoWriter("Videos/burger_prep_demo_bbs_"+BURGER_PREP_SERIES+".avi", cv2.VideoWriter_fourcc('X','V','I','D'), 
                                fps=int(cap.get(cv2.CAP_PROP_FPS)), frameSize=(512, 512))
    print(cap.get(cv2.CAP_PROP_FPS))


FRAME_COUNT = 0
PRED_FRAME_CNT = 5
MIN_SCORE_THRESHOLD = 0.70
MAX_NUM_BOXES = 10
MAX_IOU_THRESHOLD = 0.5

lowest_score = 1
lowest_score_class = ""

while cap.isOpened():
    ret, frame = cap.read()
    if(ret):
        img_width = frame.shape[0]
        img_height = frame.shape[1]

        if VIDEO_STATUS is not None and VIDEO_STATUS == "r":
            print("Recording Video...")
            video_rec.write(frame)

        # Resize the image to the model specs
        if img_width != 512 and img_height != 512:
            frame = cv2.resize(frame, (512, 512), interpolation = cv2.INTER_AREA)
            # frame, unpadded_size, padding = uf.resize_with_aspect_ratio(frame, 512)
        
        if FRAME_COUNT % PRED_FRAME_CNT == 0:
            img_np = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            output_dict = uf.run_inference_for_single_image(model, img_np)

            detection_boxes, detection_scores, detection_classes = uf.select_best_bbs(output_dict, MIN_SCORE_THRESHOLD, MAX_NUM_BOXES, MAX_IOU_THRESHOLD)
            print("detection_boxes = {}".format(detection_boxes))
            print("detection_scores = {}".format(detection_scores))
            print("detection_classes = {}".format(detection_classes))
            if len(detection_scores) > 0:
                min_score = np.min(detection_scores)
                min_index = np.argmin(detection_scores)
                if min_score < lowest_score:
                    lowest_score = min_score
                    lowest_score_class = detection_classes[min_index]

        # draw predicted label boxes onto the image
        frame = uf.get_image_with_pred_lbl_box(frame, detection_boxes, detection_scores, detection_classes, label_map)

        # show image
        cv2.imshow('video', frame)

        if VIDEO_STATUS is not None and VIDEO_STATUS == "r":
            print("Recording Video with BBS...")
            video_rec_bbs.write(frame)

        # #interpreter.set_tensor(input_details[0]['index'], [np.float32(img)])
        # interpreter.set_tensor(input_details[0]['index'], [img])

        # interpreter.invoke()
        # rects = interpreter.get_tensor(output_details[0]['index'])
        # scores = interpreter.get_tensor(output_details[1]['index'])
        # print(output_details)
        # print("For file {}".format(filename))
        # print("Rectangles are: {}".format(rects))
        # print("Scores are: {}".format(scores))
        
        FRAME_COUNT += 1
        key_pressed = cv2.waitKey(1)
        if key_pressed == ord('q'):
            break
    else:
        break

print("Frame Count = {}".format(FRAME_COUNT))
print("Min score = {}, Label = {}".format(lowest_score, label_map.get(lowest_score_class)))
cap.release()
if VIDEO_STATUS is not None and VIDEO_STATUS == "r":
    video_rec.release()
    video_rec_bbs.release()
cv2.destroyAllWindows()