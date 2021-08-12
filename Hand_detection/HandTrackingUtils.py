import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_hands=2, detectionConf=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.detectionCon = detectionConf
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    if id % 4 == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    detector = handDetector(max_hands=1)

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)

        lmList = detector.findPosition(img)
        print(lmList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()