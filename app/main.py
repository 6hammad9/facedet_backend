import datetime
# import tensorflow as tf

# tf.config.set_visible_devices([],'GPU')
import time

from line_profiler_pycharm import profile
from waitress import serve
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

from OCR.face2 import embedding_distance, Detections
import os
import cv2
import requests
import pandas as pd
import jsonpickle
import numpy as np
import cv2
import requests
import json

from norfair import OptimizedKalmanFilterFactory, Tracker

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload')
os.path.exists(BASE_PATH)



# client
@app.route('/archiving', methods=['POST', 'GET'])
def archive_data():
    test_url = 'http://localhost/emacsPanel/API/pushData.php'
    headers = {'content-type': 'image/jpeg'}
    category, person_name, img = getImages()
    # img = cv2.imread(r'E:\pytorch-flask-api\OCR\data2\imran2.PNG')
    _, img_encoded = cv2.imencode('.jpg', img)
    body = jsonpickle.encode({'cam_id_req': '4568ErrtFLli3s298zZVv',
                              'cam_id': 1,
                              'findings': category,
                              'personname': person_name,
                              'datetime': datetime.datetime.now().isoformat(),
                              'fileToUpload': img_encoded.tobytes()})
    response = requests.post(test_url, data=body, headers=headers)

    return json.loads(response.text)


# server
# app = Flask(__name__)
# @app.route('/api/test', methods=['POST'])
# def test():
#     r = request
#     # convert string of image data to uint8
#     nparr = np.frombuffer(r.data, np.uint8)
#     # decode image
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # build a response dict to send back to client
#     response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
#     response = {'message': 234}
#     # encode response using jsonpickle
#     response_pickled = jsonpickle.encode(response)
#     return Response(response=response_pickled, status=200, mimetype="application/json")
# app.run(host="0.0.0.0", port=5000)


@app.route('/getdata', methods=['GET'])
def data():
    df = getDataFromFrame()
    print("pritning again")
    print(df)
    return df.to_json()
    # return jsonify({'name': 'name',
    #                 'address': 'address'})


# import the necessary packages
from threading import Thread
import sys
import cv2
from queue import Queue


# q = asyncio.Queue(100)


class FileVideoStream:
    def __init__(self, path,c, queueSize=33):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.camid = c
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.tracker = Tracker(
            initialization_delay=3,
            distance_function="euclidean",
            hit_counter_max=35,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=50,
            past_detections_length=5,
            reid_distance_function=embedding_distance,
            reid_distance_threshold=0.92,
            reid_hit_counter_max=500,
        )
        self.detect = Detections(320, 320, 320, 320, )

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

        return self


    @profile
    def update(self):
        while not self.stopped:
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                result = self.detect.yolo_predictions(frame, self.tracker, self.camid)
                self.Q.put(result)
                _, img_encoded = cv2.imencode('.jpg', frame)
                # response = requests.post(self.api_url, data=img_encoded.tobytes(),
                # params={'cam_id_req': '4568ErrtFLli3s298zZVv'})

                # if response.status_code != 200:
                # print(f"Error sending frame to API: {response.text}")
                # else:
                # time.sleep(0.1)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

@profile
def gen_frames(url, camid):
    total_time = time.perf_counter()
    t = FileVideoStream(url,camid).start()
    # t = FileVideoStream(r'E:\pc2\1.avi',1).start()
    # t = FileVideoStream('rtsp://admin:admin123@192.168.1.15:554/cam/realmonitor?channel=1&subtype=0').start()



    # todo, this time....
    time.sleep(.5)
    print('generate frames')
    # f = Thread(target=frames, args=())
    # f.daemon = True
    # f.start()
    # frames()23
    # t.tracker.tracked_objects
    tt= 0
    i=0
    while True:
        start_time = time.perf_counter()
        if t.more():

            result = t.read()

            # result = yolo_predictions(frame)
            # if not q.full():
            #     q.put(result)
            # ret, buffer = cv2.imencode('.jpg', result)
            # frame = buffer.tobytes()
            # result = q.get()
            ret, buffer = cv2.imencode('.jpg', result)
            frame = buffer.tobytes()

            end_time = time.perf_counter()
            # print('frames: ', end_time - start_time)
            ti= 1 / (end_time - start_time)
            tt+=ti
            i+=1
            # print("fps",ti,"\n\n")
            # print("avg",tt/i,"\n\n")

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # print('ququ is empty now')
            time.sleep(0.001)
        end_time = time.perf_counter()
        # print('fps: ', end_time - start_time)
        # frame = asyncio.run(frames())
        # print("\n\n", 1 / (end_time - start_time))
        # print('__________________')
        # ret, buffer = cv2.imencode('.jpg', result)
        # frame = buffer.tobytes()
        # end_time = time.perf_counter()



        # print('fps: ', end_time - start_time)
        # print("\n\n", 1 / (end_time - start_time))
        # print('\n\n__________________')



        # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    end_time = time.perf_counter()
    # print('Total_time: ', end_time - total_time)

# @app.route('/video_feed/<url>')
# def video_feed(url):
#     print(url)
#
#     return Response(gen_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame', )

@app.route('/video_feed/<url>')
def video_feed(url):
    print(url)
    return Response(gen_frames(f'rtsp://admin:ailab425@192.168.1.108/cam/realmonitor?channel={url}&subtype=0',url), mimetype='multipart/x-mixed-replace; boundary=frame', )
    #http://127.0.0.1:6033/video_feed/1.avi link to run video feed
# @app.route('/video_feed/<url>')
# def video_feed(url):
# test_url = 'http://localhost/emacsPanel/api/getConnected.php'
# # headers = {'content-type': 'text/html'}
# # body = jsonpickle.encode({'cam_id_req': '4568ErrtFLli3s298zZVv'})
# body = {'cam_id_req': '4568ErrtFLli3s298zZVv'}
# # response = requests.post(test_url, data=body, headers=headers)
# res = requests.post(test_url, data=body)
# print(res.text)
# #     return Response(gen_frames(f'rtsp://admin:ailab425@192.168.1.108/cam/realmonitor?channel={url}&subtype=0'), mimetype='multipart/x-mixed-replace; boundary=frame', )

@app.route('/')
def index():
    return render_template('yolo.html')


def start():
    # start a thread to read frames from the file video stream
    t = Thread(target=queryCam, args=())
    t.daemon = True
    t.start()



def queryCam():
    # keep looping infinitely
    while True:
        # if the thread indicator variable is set, stop the
        # thread
        test_url = 'http://localhost/emacsPanel/api/getConnected.php'
        # headers = {'content-type': 'text/html'}
        # body = jsonpickle.encode({'cam_id_req': '4568ErrtFLli3s298zZVv'})
        body = {'cam_id_req': '4568ErrtFLli3s298zZVv'}
        # response = requests.post(test_url, data=body, headers=headers)
        res = requests.post(test_url, data=body)
        if res.status_code==200 and res.text != '[]':
            for cam in json.loads(res.text):
                print(cam['channelNumber'])
                print(cam['camid'])
                # gen_frames(f"rtsp://admin:ailab425@192.168.1.108/cam/realmonitor?channel={cam['channelNumber']}&subtype=0",cam['camid'])
                test_url = 'http://localhost/emacsPanel/api/geturl.php'
                # headers = {'content-type': 'text/html'}
                # body = jsonpickle.encode({'cam_id_req': '4568ErrtFLli3s298zZVv'})
                body = {
                    'cam_id_req': '4568ErrtFLli3s298zZVv',
                    'urlstring': f"http://localhost:6033/video_feed/{cam['channelNumber']}",
                    'camid': cam['camid'],
                        }
                # response = requests.post(test_url, data=body, headers=headers)
                #res = requests.post(test_url, data=body)
                #print(res.text)
        else:
            continue
        # print(res.text)
        time.sleep(1)



start()

if __name__ == "__main__":
    app.run(debug=False, port=6033, threaded=True,host='0.0.0.0')
