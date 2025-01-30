import os
import cv2
import sys
import mediapipe as mp
import VisualizeObjects as vo
import Utility_Functions as uf
from subprocess import PIPE, run
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BURGER_PREP_SERIES = "1"
if len(sys.argv) > 1:
    BURGER_PREP_SERIES = sys.argv[1]

# Uncomment this block of code if you want to create the image frames from a live video camera
# and comment out the section below that creates the images from a prerecorded video
"""
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

cap = cv2.VideoCapture(device_id)
"""

# Uncomment the line of code below if you want to create the image frames from prerecorded video.
# Comment out the section above that creates the images from a live video camera
cap = cv2.VideoCapture("Videos/burger_prep_demo_"+BURGER_PREP_SERIES+".avi")

if not cap.isOpened():
    print("No stream :(")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
img_width = int(cap.get(3))
img_height = int(cap.get(4))
# print("FPS = {}, Width = {}, Height = {}".format(fps, width, height))

# Uncomment this block below if you are creating the image frames from a live video stream 
# and want to save the video
"""
video_rec = cv2.VideoWriter("Videos/burger_prep_"+BURGER_PREP_SERIES+".avi",
                    cv2.VideoWriter_fourcc('X','V','I','D'),
                    fps=fps, frameSize=(width,height))
"""

FRAME_COUNT = 0
SAMPLE_RATE = 10
img_folder = 'Images'
output_folder = os.path.join('.', img_folder)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

while cap.isOpened():
    ret, frame = cap.read()
    
    key_pressed = cv2.waitKey(10)

    # This piece of code saves the frame from the video as an image when the 'p' button is pressed
    if (FRAME_COUNT % SAMPLE_RATE == 0) or (key_pressed == ord('p')):
        img_file_name = 'burger_'+BURGER_PREP_SERIES+'_img_'+str(FRAME_COUNT)+'.jpg'
        cv2.imwrite('./'+img_folder+'/'+img_file_name, frame)

    cv2.imshow('Video', frame)

    FRAME_COUNT += 1
    
    # Quit the progrem if the user enters 'q'
    if key_pressed == ord('q'):
        break

print("Frame Count = {}".format(FRAME_COUNT))
cap.release()
# video_rec.release()
cv2.destroyAllWindows()