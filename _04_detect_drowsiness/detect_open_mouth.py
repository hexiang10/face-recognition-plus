
# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
# 按下'q'键退出程序

# 声音问题出现：指定的设备未打开，或不被 MCI 所识别。
# 解决：修改playsound.py文件，将这行：command = ' '.join(command)   # .encode('utf-16') 注释掉这个

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import dlib
import cv2
import imutils
import time

# 设置文件路径
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', type=str, default="shape_predictor_68_face_landmarks.dat",
                help='path to the facial landmark predictor')
ap.add_argument('-w', '--webcam', type=int, default=0,
                help='index of webcam on system')
args = vars(ap.parse_args())

MOUTH_AR_THRESH = 0.79

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

print("[INFO]loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

(mStart, mEnd) = (49,68)

print("[INFO]starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)

    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame,[mouthHull],-1,(0,255,0),1)
        cv2.putText(frame, 'MRA:{:.2f}'.format(mar), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame,'mouth is open!',(30,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()

