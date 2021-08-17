from Utils import HandTrackingUtils
import cv2
import mediapipe as mp
import os


# Wcam, Hcam = 480, 640
# cap.set(3, Wcam)
# cap.set(4, Hcam)

def mapping(s):
    return s.count('1')


finger_imgs = []
folder = "finger_counting_imgs"
pwd = os.listdir(folder)
pwd.sort()

finger_list = []
for img_path in pwd:
    finger_img = cv2.imread(f'{folder}/{img_path}')
    finger_list.append(finger_img)

TIPS = [4, 8, 12, 16, 20]

if __name__ == '__main__':
    handDetector = HandTrackingUtils.HandDetector(max_hands=1, detectionConf=0.7, trackCon=0.7)
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        img = handDetector.findHands(img, draw=True)
        lmList = handDetector.findPosition(img)

        fingers = ['0', '0', '0', '0', '0']
        if len(lmList):

            # thumb

            # left and right TODO: front side and back side check
            left = (lmList[TIPS[0]][1] < lmList[TIPS[1]][1])
            right = (lmList[TIPS[0]][1] > lmList[TIPS[1]][1])

            if lmList[TIPS[0]][1] < lmList[TIPS[0] - 1][1]:
                fingers[0] = '1'

            for idx in range(1, 5):
                if lmList[TIPS[idx]][2] < lmList[TIPS[idx] - 2][2]:
                    fingers[idx] = '1'
                else:
                    fingers[idx] = '0'

            finger_count = fingers.count('1')
            h, w, c = finger_list[finger_count].shape
            img[0:h, 0:w] = finger_list[finger_count]

            # lable finger count
            cv2.rectangle(img, (20, 225), (170, 425), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, str(finger_count), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10,
                        (0, 0, 0), 25)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
