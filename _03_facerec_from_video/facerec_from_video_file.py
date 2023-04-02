import face_recognition
import cv2
import numpy as np

# 读入影片并得到影片长度
input_movie = cv2.VideoCapture("Huanlesong.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

'''
第一个参数是要保存的文件路径
fourcc 指定编码器
fps 要保存视频的帧率
frameSize 要保存的文件的画面尺寸
isColor 指示是黑白画面还是彩色画面
fourcc本身是一个32位的无符号数值，用4个字母表示采用的编码器
常用的有"DIVX","MJPG","XVID","X264"
推荐使用"XVID"，但一般依据你的电脑环境安装了哪些编码器

'''

fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_movie = cv2.VideoWriter('output.avi', fourcc, 25, (1280, 720))

andi_image = face_recognition.load_image_file("andi.jpg")
andi_face_encoding = face_recognition.face_encodings(andi_image)[0]

quxiaoxiao_image = face_recognition.load_image_file("quxiaoxiao.jpg")
quxiaoxiao_face_encoding = face_recognition.face_encodings(quxiaoxiao_image)[0]

shuzhan_image = face_recognition.load_image_file("shuzhan.jpg")
shuzhan_face_encoding = face_recognition.face_encodings(shuzhan_image)[0]

know_faces = [
    andi_face_encoding,
    quxiaoxiao_face_encoding,
    shuzhan_face_encoding
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(
            know_faces, face_encoding, tolerance=0.5)
        name = None
        if match[0]:
            name = "andi"
        elif match[1]:
            name = "quxiaoxiao"
        else:
            name = "shuzhan"

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom-25),(right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6),
                    font, 0.5, (255, 255, 255), 1)

    print("writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()

