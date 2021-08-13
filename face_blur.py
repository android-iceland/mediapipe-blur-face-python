import streamlit as st
import shutil
import cv2
import mediapipe as mp
import os
from os import path

st.title("Face blur")
try:
    os.mkdir("temp")
except:
    pass
for i in os.listdir("./temp/"):
    try:
        os.remove(os.remove(f"./temp/{i}"))
    except:
        pass
input_file_path = ""
uploaded_file = st.file_uploader("Upload Files", type=["mp4"])
if uploaded_file is not None:
    with open(f"./temp/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    input_file_path = f"./temp/{uploaded_file.name}"


folder_path = st.text_input("Paste the folder location where you want to save:")
flag = path.exists(folder_path)


class FaceDetector:
    def __init__(self, minDetectionCon=0.45,pixel_size=0.05):
        self.pixel_size=pixel_size
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw) - 40,
                    int(bboxC.ymin * ih) - 150,
                    int(bboxC.width * iw) + 80,
                    int(bboxC.height * ih) + 200,
                )
                x, y, w, h = bbox
                x1, y1 = x + w, y + h
                try:
                    face_img = img[y:y1, x:x1].copy()
                    face_img = cv2.resize(
                        face_img,
                        dsize=None,
                        fx=self.pixel_size,
                        fy=self.pixel_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    face_img = cv2.resize(
                        face_img,
                        dsize=(x1 - x, y1 - y),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    img[y:y1, x:x1] = face_img
                except:
                    pass
        return img


def main(detectionConfidence,blur_size,flip_the_video):
    global folder_path
    global input_file_path
    FRAME_WINDOW = st.image([])
    input_file = input_file_path
    cap = cv2.VideoCapture(input_file)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    size = (width, height)
    file_name = "./temp/output.mp4"
    if folder_path.endswith("/"):
        export_file_path = f"{folder_path}"
    else:
        export_file_path = f"{folder_path}/"

    var1 = os.system(f'ffmpeg -i {input_file} "./temp/audio.mp3"')

    if var1 == 0:
        print("audio extracted")
    # codec = cv2.VideoWriter_fourcc(*"mpeg")
    codec = cv2.VideoWriter_fourcc(*"MP4V")
    video_output = cv2.VideoWriter(file_name, codec, framerate, size, True)

    detector = FaceDetector(minDetectionCon=float(detectionConfidence / 100),pixel_size=float((blur_size)/1000))
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    while ret:

        success, img = cap.read()
        if flip_the_video =="Yes":
            img = cv2.flip(img, 1)
        elif flip_the_video == "No":
            pass
        try:
            img = detector.findFaces(img, True)
        except Exception as e:
            print(f"error is {e}")
        if img is None:
            break

        video_output.write(img)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame) 
    video_output.release()
    cap.release()
    aduio_file = "./temp/audio.mp3"
    blur_video = "./temp/blur.mp4"
    os.system(
        f"ffmpeg -i {file_name} -i {aduio_file} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {blur_video}"
    )
    rename_file_name = f"{export_file_path}output_" + input_file.split("/")[-1]
    try:
        os.remove(rename_file_name)
    except:
        pass
    shutil.copy(blur_video, rename_file_name)


if __name__ == "__main__":
    detectionConfidence = st.slider("Face Detection Confidence")
    blur_size = st.slider("Blur pixel size in face")
    flip_the_video = st.selectbox("Horizontally flip video ",("Yes","No"))
    if st.button("Start Face Blur"):
        if flag:
            main(detectionConfidence,blur_size,flip_the_video)
            st.markdown(f"## Face blur complete check your export folder")

            for i in os.listdir("./temp/"):
                try:
                    os.remove(os.remove(f"./temp/{i}"))
                except:
                    pass
        else:
            st.error("Export folder not exist.")
