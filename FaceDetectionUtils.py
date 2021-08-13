import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionConf=0.5):
        self.minDectionCon = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # --- print bounding box with key points ---
                # mpDraw.draw_detection(img, detection)
                # --- print bounding box with key points ---

                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                cv2.rectangle(img, bbox, (0, 0, 255), 2)

                # write the confidence of a face
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1] - 20), cv2.FONT_ITALIC, 3, (0, 0, 255), 2)
        return img, bboxs

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()