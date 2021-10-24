# mediapipe blur face python

## Click on the video to watch on youtube
 [![Watch the video](https://github.com/android-iceland/mediapipe-blur-face-python/blob/main/demo/demo.gif)](https://youtu.be/SssGcxpKZTg)

<br>
This web app only run on localhost <br>
Python version >=3.7 <br>


# If you don't know how to install python
Python installation tutorial:<br>
https://youtu.be/UBsONplrOH4  <br>
 
Install pip:(If you got an error pip not install )  <br>
https://phoenixnap.com/kb/install-pip-windows  <br>


## step 1
Open cmd or terminal and paste this line
```
pip install -r requirements.txt
```
## Run web app
To run the web app  or, double click on the run.py file

```
python run.py
```

![Demo](https://github.com/android-iceland/mediapipe-blur-face-python/blob/main/demo/webapp.PNG)
#### Give a clear input file name like "video.mp4" not like "vid - 324$op _ 9363*.mp4"
#### If you increase "Face Detection Confidence" then face tracking will more accurate. <br>
#### If you reduce "Blur pixel size in face" then face blur pixel size will increase. <br>
#### In "Horizontally flip video" "yes" mean you can mirror your video. "No" mean nothing will happen.


## From webcam
Realtime face blur using webcam
```
python webcam.py
```


# Blur face and Change background
```
streamlit run face_blur_and_background_change.py --server.maxUploadSize=5000
```
![Demo](https://github.com/android-iceland/mediapipe-blur-face-python/blob/main/demo/test.png)

# Blur face and Change background
```
python face_block_and_background_change_webcam.py
```
