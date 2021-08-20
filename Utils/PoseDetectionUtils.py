import cv2
import mediapipe as mp
import math
import numpy as np


class PoseDetector:
    def __init__(self, mode=False, upBody=0, smooth=True, detectConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectConf
        self.trackingCon = trackingConf

        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.upBody, smooth_landmarks=self.smooth,
        #                              min_detection_confidence=self.detectCon, min_tracking_confidence=self.trackingCon )

        self.pose = self.mpPose.Pose(min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append((id, cx, cy))
                if draw and id > 10:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]

        # Calculate Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        bar = np.interp(angle, (220, 310), (0, 50))
        bar_sign = p1 % 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)

            # cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDist(self, img, p1, p2):
        pass
