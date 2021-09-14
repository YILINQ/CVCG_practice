from Utils.PoseDetectionUtils import PoseDetector
import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller
from PIL import ImageGrab
import keyboard

from AppKit import NSScreen

# print(NSScreen.mainScreen().frame().size.width)
# print(NSScreen.mainScreen().frame().size.height)

# 1792 1120
# step 1: capture screen, pass info to model
# step 2: take in image and detect pose
# step 3: control mouse and fire

# {(450, 250), (1450, 850)}


# center box of aiming frame

mouse_er = Controller()
if __name__ == '__main__':
    # step 1
    PoseDetector = PoseDetector(detectConf=0.3)
    while True:
        img = ImageGrab.grab(bbox=(900, 500, 2700, 1620))
        img_np = np.array(img)

        img_test = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        pose_img = PoseDetector.findPose(img_test, draw=True)
        # landmark_list = PoseDetector.getPosition(img_test, draw=False)

        if PoseDetector.results.pose_landmarks:
            landmark_list = PoseDetector.getPosition(img_test, draw=False)

            if landmark_list != []:
                # head found, just shoot it
                print("FOUND")
                mouse_er.position = (landmark_list[0][1], landmark_list[0][2])
                mouse_er.press(Button.left)

                mouse_er.release(Button.left)

        cv2.imshow('test', pose_img)
        cv2.waitKey(1)

        # step 3
        # mouse_er.position
        # mouse_er.position = (new pos)
        # mouse_er.press(Button.left)
        # mouse_er.release(Button.left)
        # time.sleep(gap)
