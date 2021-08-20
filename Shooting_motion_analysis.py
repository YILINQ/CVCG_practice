from Utils.PoseDetectionUtils import PoseDetector
import cv2
import mediapipe as mp

# cap = cv2.VideoCapture('zihao_back.mp4')
cap = cv2.VideoCapture('Tompson_back.mp4')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def draw_bar(img, starting_pos, ending_pos, angle, text=None):
    bar = int((360 - angle) / 360 * 100)
    if int((360 - angle) / 360 * 100) >= 99:
        bar = 100
    cv2.rectangle(img, starting_pos, ending_pos, (0, 0, 255))
    cv2.rectangle(img, (starting_pos[0], bar), ending_pos, (0, 0, 255), cv2.FILLED)
    cv2.putText(img, text, (starting_pos[0], ending_pos[1] + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(img, f'{int(int((360 - angle) / 360 * 100))}%', (starting_pos[0], ending_pos[1] + 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


def main(Path):
    cap = cv2.VideoCapture(Path)
    detector = PoseDetector()
    # img = cv2.imread('./shooting_motions_test/Jerry_shooting.jpg')
    while True:
        success, img = cap.read()

        img = detector.findPose(img, draw=False)
        lmList = detector.getPosition(img, draw=False)
        left_arm = detector.findAngle(img, 11, 13, 15)
        right_arm = detector.findAngle(img, 12, 14, 16)
        left_leg = detector.findAngle(img, 23, 25, 27)
        right_leg = detector.findAngle(img, 24, 26, 28)

        # analysis of left right arms and legs

        # # left arm
        draw_bar(img, (80, 0), (125, 100), left_arm, 'la')
        draw_bar(img, (160, 0), (205, 100), right_arm, 'ra')
        draw_bar(img, (240, 0), (285, 100), left_leg, 'lg')
        draw_bar(img, (300, 0), (345, 100), right_leg, 'rg')

        print(right_arm)

        # print([left_arm, right_arm, left_leg, right_leg])
        cv2.imshow("Image", img)
        cv2.waitKey(1000)


if __name__ == "__main__":
    main('zihao_back.mp4')
