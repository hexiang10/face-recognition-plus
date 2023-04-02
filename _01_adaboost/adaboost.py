# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
import cv2

# 读取一张图片
img = cv2.imread('data/00.jpg')
# 将图片转化为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 此文件是opencv的haar人脸特征分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# 一定要告诉编译器文件所在的位置
face_cascade.load('haarcascade_frontalface_alt2.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()
