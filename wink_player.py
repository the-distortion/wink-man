import numpy as np
import cv2
import dlib
import keyboard, time

from EAR import eye_aspect_ratio

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 6

COUNTER_LEFT = []
TOTAL_LEFT = 0

COUNTER_RIGHT = []
TOTAL_RIGHT = 0

EYES_CLOSED = []

def main():
    global COUNTER_LEFT, COUNTER_RIGHT, TOTAL_LEFT, TOTAL_RIGHT, EYES_CLOSED
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    video_capture = cv2.VideoCapture(0) #for webcam

    while True:
        ret, frame = video_capture.read()

        if ret: #if available
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale
            rects = detector(gray, 0)

            for rect in rects:
                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()

                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                cv2.putText(frame, "E.A.R. Left : {:.2f}".format(ear_left),\
                    (300, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 150, 255), 2)
                cv2.putText(frame, "E.A.R. Right: {:.2f}".format(ear_right),\
                    (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)

                EYES_CLOSED = []

                if ear_left < EYE_AR_THRESH:
                    COUNTER_LEFT.append(ear_left)
                else:
                    if len(COUNTER_LEFT) >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_LEFT += 1  
                        print("Left eye winked", (COUNTER_LEFT));
                        keyboard.write(' ') #toggling play/pause
                        time.sleep(0.5)
                        EYES_CLOSED.append(sum(COUNTER_LEFT)/len(COUNTER_LEFT))
                    COUNTER_LEFT = []

                if ear_right < EYE_AR_THRESH:  
                    COUNTER_RIGHT.append(ear_right)
                else:  
                    if len(COUNTER_RIGHT) >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_RIGHT += 1  
                        print("Right eye winked", (COUNTER_RIGHT));
                        keyboard.write(' ') #toggling play/pause
                        time.sleep(0.5)
                        EYES_CLOSED.append(sum(COUNTER_RIGHT)/len(COUNTER_RIGHT))
                    COUNTER_RIGHT = []

                if len(EYES_CLOSED)>=2 and abs(EYES_CLOSED[0] - EYES_CLOSED[1]) <= 0.035:
                    print("exitting\nEAR left, EAR right =", EYES_CLOSED);return
                else: EYES_CLOSED=[]

            cv2.putText(frame, f"Wink Left : {TOTAL_LEFT}", (10, 30),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            cv2.putText(frame, f"Wink Right: {TOTAL_RIGHT}", (10, 60),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

            cv2.imshow("Faces found", frame)

        ch = 0xFF & cv2.waitKey(1)
   
        if ch == ord('q'): break

main()
cv2.destroyAllWindows()