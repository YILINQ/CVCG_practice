from Utils.PoseDetectionUtils import PoseDetector
import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller
from PIL import ImageGrab
from

# step 1: capture screen, pass info to model
# step 2: take in image and detect pose
# step 3: control mouse and fire

# {(450, 250), (1450, 850)}


# center box of aiming frame
BOUNDING_BOX = {'left': 450, 'top': 250, 'width': 1000, 'height': 600}

mouse_er = Controller()
if __name__ == '__main__':
    # step 1
    PoseDetector = PoseDetector()
    while True:
        img = ImageGrab.grab(bbox=(0, 0, 1280, 720))
        img_np = np.array(img)
        cv2.imshow('test', img_np)
        cv2.waitKey(1)

        # step 3
        # mouse_er.position
        # mouse_er.position = (new pos)
        # mouse_er.press(Button.left)
        # mouse_er.release(Button.left)
        # time.sleep(gap)
