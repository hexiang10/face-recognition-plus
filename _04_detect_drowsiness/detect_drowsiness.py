
# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
# 按下'q'键退出程序

# 声音问题出现：指定的设备未打开，或不被 MCI 所识别。
# 解决：修改playsound.py文件，将这行：command = ' '.join(command)   # .encode('utf-16') 注释掉这个

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import dlib
import cv2
import imutils
import time

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 设置文件路径
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', type=str, default="shape_predictor_68_face_landmarks.dat",
                help='path to the facial landmark predictor')
ap.add_argument('-a', '--alarm', type=str, default="alarm.wav",
                help='path to the alarm .WAV file')
ap.add_argument('-w', '--webcam', type=int, default=0,
                help='index of webcam on system')
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CLOSE_FRAMES = 48
COUNTER = 0
ALARM_ON = False

print("[INFO]loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

print("[INFO]starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)

    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER +=1
            if COUNTER >= EYE_AR_CLOSE_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                    if args['alarm'] != "":
                        # alarm 为音频文件名
                        t = Thread(target=sound_alarm,args=(args['alarm'],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame,'DROWSINESS ALARM!',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, 'ERA:{:.2f}'.format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()

