import cv2
import mediapipe as mp
from FaceDetectionUtils import FaceDetector
from HandTrackingUtils import HandDetector


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    HD = HandDetector(max_hands=2)
    FD = FaceDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = HD.findHands(img, draw=True)
        img, bboxs = FD.findFaces(img, draw=True)
        lmList = HD.findPosition(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
