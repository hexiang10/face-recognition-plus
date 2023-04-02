from PIL import Image, ImageDraw
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake
# pip install dlib-19.23.0-cp39-cp39-win_amd64.whl （python 3.9）
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple face_recognition
import face_recognition

image = face_recognition.load_image_file('obama.jpg')

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # 画个浓眉
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(1, 1, 1, 1), width=15)
    # d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150),width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # 涂个性感嘴唇
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 128), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128), width=1)

    # 闪亮的大眼睛
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

pil_image.show()
