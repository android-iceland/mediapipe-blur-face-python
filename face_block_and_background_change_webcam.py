import cv2
import mediapipe as mp
import numpy as np
import time



class FaceDetector:
    def __init__(self, minDetectionCon=0.45):
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
                #add some value to increase the box size
                bbox = (
                    #play with the values to change the blur box area
                            int(bboxC.xmin * iw) - 40,
                            int(bboxC.ymin * ih) - 100,
                            int(bboxC.width * iw) + 80,
                            int(bboxC.height * ih) + 150,
                )
                #play with the value to increase box size
                x, y, w, h = bbox
                x1, y1 = x + w, y + h
                try:
                    face_img = img[y:y1, x:x1].copy()
                    face_img = cv2.resize(
                        face_img,
                        dsize=None,
                        fx=0.05, #decrease the value you will get big pixel blur
                        fy=0.05,  #decrease the value you will get big pixel blur
                        interpolation=cv2.INTER_NEAREST,
                    )
                    face_img = cv2.resize(
                        face_img,
                        dsize=(x1 - x, y1 - y),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    img[y:y1, x:x1] = face_img
                except Exception as e:
                    print(e)
                    pass
        return img




class BackgroundRemove:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation=self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    def remove_background(self ,img,option,original_background):
        if option == "color":
            BG_COLOR = (0, 255, 0)
        
        bg_image = None
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        condition = np.stack((self.results.segmentation_mask,) * 3, axis=-1) > 0.1
        if bg_image is None:
            if option=="blur":
                bg_image = cv2.GaussianBlur(image,(55,55),0)
            elif option == "color":
                BG_COLOR = (0, 255, 0)
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
            elif option=="path":
                ih, iw, ic = image.shape
                bg_image = original_background
                bg_image=cv2.resize(bg_image, (iw,ih))
                # bg_image = cv2.GaussianBlur(bg_image,(55,55),0)
        output_image = np.where(condition, image, bg_image)
        return output_image

def main():
    cap = cv2.VideoCapture("video.mp4")
    # cap = cv2.VideoCapture(0)
    pTime = 0
    background_remove_object = BackgroundRemove()
    detector = FaceDetector()
    original_background=cv2.imread('img.jpg')
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        # option="color"
        # option="blur"
        option="path"
        img = background_remove_object.remove_background(img,option,original_background)
        img = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(
        #     img,
        #     f"FPS : {int(fps)}",
        #     (20, 70),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     3,
        #     (0, 255, 0),
        #     2,
        # )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()





