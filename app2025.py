import datetime
import time
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import os
import cv2
import requests
import pandas as pd
import jsonpickle
import numpy as np
from threading import Thread, Lock
from queue import Queue
from dotenv import load_dotenv

# Our imports
from OCR.face2025old import embedding_distance, Detections
from norfair import OptimizedKalmanFilterFactory, Tracker

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload')
os.path.exists(BASE_PATH)

# Configuration
NODE_API_URL = os.getenv('NODE_API_URL', 'http://localhost:3000/api/cameras')
STREAM_TIMEOUT = int(os.getenv('STREAM_TIMEOUT', '5'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

# Thread-safe storage for active streams
active_streams = {}
stream_lock = Lock()

class FileVideoStream:
    def __init__(self, cam_id, queueSize=33):
        self.cam_id = cam_id
        self.stream = None
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.lock = Lock()
        self.last_frame_time = time.time()
        
        # Initialize tracker and detector
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
        self.detect = Detections(320, 320, 320, 320)
        
        self.initialize_stream()

    def initialize_stream(self):
        """Initialize the video stream based on camera configuration"""
        try:
            # Get camera config from Node.js API
            response = requests.get(
                f"{NODE_API_URL}/by-id/{self.cam_id}",
                timeout=STREAM_TIMEOUT
            )
            
            if response.status_code != 200:
                raise ValueError(f"Camera {self.cam_id} not found in database")
                
            cam_config = response.json()
            print(f"Initializing camera {self.cam_id} with config: {cam_config}")

            # Handle different stream types
            if cam_config['stream_type'] == 'local':
                # Local camera (webcam)
                self.stream = cv2.VideoCapture(
                    int(cam_config['stream_source']),
                    cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
                )
            elif cam_config['stream_type'] == 'rtsp':
                # RTSP stream
                self.stream = cv2.VideoCapture(cam_config['stream_source'])
            elif cam_config['stream_type'] == 'http':
                # HTTP stream
                self.stream = cv2.VideoCapture(cam_config['stream_source'])
            else:
                raise ValueError(f"Unknown stream type: {cam_config['stream_type']}")

            if not self.stream.isOpened():
                raise ValueError(f"Failed to open video source: {cam_config['stream_source']}")

            # Set basic properties
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.cam_id} initialized successfully")

        except Exception as e:
            print(f"Error initializing camera {self.cam_id}: {str(e)}")
            self.stopped = True
            raise

    def start(self):
        """Start the thread to read frames from the video stream"""
        if self.stopped:
            return self

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Thread that continuously reads frames from the stream"""
        while not self.stopped:
            try:
                with self.lock:
                    grabbed, frame = self.stream.read()
                    self.last_frame_time = time.time()

                if not grabbed:
                    print(f"Camera {self.cam_id}: No frame grabbed")
                    time.sleep(0.1)
                    continue

                # Process frame with detection
                result = self.detect.yolo_predictions(frame, self.tracker, self.cam_id)
                
                if not self.Q.full():
                    self.Q.put(result)
                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Camera {self.cam_id} stream error: {str(e)}")
                time.sleep(1)
                self.reconnect()

    def reconnect(self):
        """Attempt to reconnect to the video source"""
        print(f"Attempting to reconnect camera {self.cam_id}")
        with self.lock:
            if self.stream:
                self.stream.release()
            self.initialize_stream()

    def read(self):
        """Get the next frame from the queue"""
        return self.Q.get()

    def more(self):
        """Check if there are frames in the queue"""
        return not self.Q.empty()

    def stop(self):
        """Stop the stream and release resources"""
        self.stopped = True
        with self.lock:
            if self.stream:
                self.stream.release()
        print(f"Camera {self.cam_id} stream stopped")

def get_stream(cam_id):
    """Get or create a video stream for the given camera ID"""
    with stream_lock:
        if cam_id not in active_streams:
            print(f"Creating new stream for camera {cam_id}")
            active_streams[cam_id] = FileVideoStream(cam_id).start()
            time.sleep(0.5)  # Allow time for initialization
        return active_streams[cam_id]

def cleanup_streams():
    """Periodically clean up inactive streams"""
    while True:
        time.sleep(60)
        with stream_lock:
            to_remove = []
            for cam_id, stream in active_streams.items():
                if stream.stopped or (time.time() - stream.last_frame_time > 30):
                    stream.stop()
                    to_remove.append(cam_id)
            
            for cam_id in to_remove:
                del active_streams[cam_id]
                print(f"Removed inactive stream for camera {cam_id}")

def gen_frames(cam_id):
    """Generate frames from the specified camera"""
    try:
        stream = get_stream(cam_id)
        
        while True:
            if stream.more():
                result = stream.read()
                ret, buffer = cv2.imencode('.jpg', result)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
                
    except Exception as e:
        print(f"Stream error for camera {cam_id}: {str(e)}")
        with stream_lock:
            if cam_id in active_streams:
                active_streams[cam_id].stop()
                del active_streams[cam_id]
        raise

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    """Video streaming route"""
    try:
        return Response(gen_frames(cam_id),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# @app.route('/video_feed/<url>')
# def video_feed(url):
#     print("\Printing URL")
#     print(url)
#     print("Printing URL/")
#     channel = 1
#     writeurl = 'rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel='+str(url)+'&subtype=0'
#     #t = FileVideoStream('rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0', 1).start()
#     return Response(gen_frames(writeurl,url), mimetype='multipart/x-mixed-replace; boundary=frame', )
#     # my comment
#     #return Response(gen_frames(f'rtsp://admin:ailab425@192.168.1.108/cam/realmonitor?channel={url}&subtype=0',url), mimetype='multipart/x-mixed-replace; boundary=frame', )

# #testing
@app.route('/camera_status/<cam_id>')
def camera_status(cam_id):
    """Check camera status"""
    try:
        stream = get_stream(cam_id)
        with stream.lock:
            return jsonify({
                "status": "active" if not stream.stopped else "inactive",
                "frame_size": f"{stream.stream.get(3)}x{stream.stream.get(4)}",
                "fps": stream.stream.get(5),
                "queue_size": stream.Q.qsize()
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_frame/<cam_id>')
def test_frame(cam_id):
    """Get a single test frame"""
    try:
        stream = get_stream(cam_id)
        if stream.more():
            frame = stream.read()
            ret, buffer = cv2.imencode('.jpg', frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        return "No frames available", 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('yolo.html')

if __name__ == "__main__":
    # Start cleanup thread
    cleanup_thread = Thread(target=cleanup_streams)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    app.run(debug=False, port=6033, threaded=True, host='0.0.0.0')