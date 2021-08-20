from Utils import PoseDetectionUtils
import cv2
import mediapipe as mp

# cap = cv2.VideoCapture('zihao_back.mp4')
cap = cv2.VideoCapture('Tompson_back.mp4')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        lmList = []
        if results:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("image", img)
    cv2.waitKey(100)
