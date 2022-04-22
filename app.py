from cv2 import CAP_DSHOW
from flask import Flask, render_template, Response, request
import cv2, queue
import threading
from threading import Lock
from Embedding_extraction import extract_emb
from MTCNN_Pytorch_camera import special_cases, draw_box, write_to_csv, recognize_faces, create_recognizer, create_detector, fail_to_grab
from MTCNN_Pytorch_camera import check_opening, load_embeddings, process_frame, process_frame_or_not
app = Flask(__name__)
from VideoCapture import VideoCapture
import pickle



# use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    embedding_list, name_list = load_embeddings(filename='classifier.pt')
    frame_counter = -1
    mtcnn = create_detector()
    resnet = create_recognizer()
    unknown_save_counter = 0
    save_counter = 0
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        frame = cam.read()  # read the camera frame
        frame_counter += 1
        if not process_frame_or_not(frame_counter):
            boxes, frame = recognize_faces(frame, resnet, mtcnn, embedding_list, name_list)
            frame_counter = 1
        ret, buffer = cv2.imencode('.jpg', frame[1])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        k = cv2.waitKey(1)
        ans = special_cases(k, frame[1])
        if ans == 0 or ans == 1:
            break
 

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route("/off-camera")
def off_camera():
    return render_template('off-camera.html')

@app.route("/on_camera")
def on_camera():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


