import cv2
import mediapipe as mp


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
        lmList = []
        if self.results:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


def main(Path):
    cap = cv2.VideoCapture(Path)
    detector = PoseDetector()
    # img = cv2.imread('./shooting_motions_test/Jerry_shooting.jpg')
    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        # lmList = detector.getPosition(img)

        cv2.imshow("Image", img)
        cv2.waitKey(100)


if __name__ == "__main__":
    main('../zihao_side_2.mp4')
