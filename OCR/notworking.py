import datetime
import cv2
import jsonpickle
import requests
from numpy.linalg import norm
import json
import os
import warnings
# from line_profiler_pycharm import profile
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy.spatial.distance import cosine
from typing import List, Optional, Union
import torch
import norfair
from norfair import Detection, Paths, Tracker, Video, OptimizedKalmanFilterFactory, get_cutout
from norfair.distances import frobenius, iou
from numpy import asarray
import time
import numpy as np
import pickle
import onnxruntime as ort
import glob


# Dynamically get the directory where face2025old.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute model paths
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov5m_dynamic.onnx")
GLINTR_PATH = os.path.join(BASE_DIR, "models", "glintr100.onnx")
F2_MODEL_PATH = os.path.join(BASE_DIR, "models", "f2.onnx")

# Define whitelist folder and pickle file path
WHITELIST_FOLDER = os.path.join(BASE_DIR, "whitelisted")
USERS_PKL_PATH = os.path.join(BASE_DIR, "users.pkl")

ort_sess = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
ort_sess_r = ort.InferenceSession(GLINTR_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
ort_sess2 = ort.InferenceSession(F2_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def load_metadata(metadata_path="D:/New folder/OCR/metadata.json"):
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}
    

metadata = load_metadata()

def get_person_status(name, camid):
    camid_str = str(camid)
    if name in metadata:
        if camid_str in metadata[name].get("cameras", []):
            return "whitelisted"
    return "notwhitelisted"
class Person:
    def __init__(self, det_time, exit_time, duration, det_history, person_name,
                 person_id, status='', f=None, p=None):
        if p is None:
            p = list()
        if f is None:
            f = list()
        self.det_time = det_time
        self.firstTime = True
        self.firstTimeName = False
        self.firstTimeNameCheck = True
        self.firstTimeNA = True
        self.exit_time = exit_time
        self.duration = duration
        self.det_history = det_history
        self.person_name = person_name
        self.person_id = person_id
        self.status = status
        self.det_face: list = f
        self.det_person: list = p
        self.db_entry_created = False
        self.last_seen = det_time
        self.entry_cooldown = False
MIN_REENTRY_SECONDS = 5

def details(self):
        return 3

    
debug = False

# Detection class instantiated in app.py
class Detections:
    # instantiate with input size the model takes, we are using 320x320 size models
    def __init__(self, wf, hf, wp, hp):
        self.WIDTH_FACE = wf
        self.HEIGHT_FACE = hf
        self.WIDTH_PERSON = wp
        self.HEIGHT_PERSON = hp
        self.person_list = []
    
    # Detects faces and persons from the provided image
    def get_detections(self, img):
        start_time = time.perf_counter()
        try:
            image = img.copy()
            row, col, d = image.shape
            # add black regions in image to preserve aspect ratio
            max_rc = max(row, col)
            input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
            input_image[0:row, 0:col] = image

            # converts the image to standard format that can be used to run inference
            blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.WIDTH_FACE, self.HEIGHT_FACE), swapRB=True,
                                         crop=False)
            blob2 = blob

            # run inference on the processed image both for face n person
            preds = ort_sess.run(None, {'images': blob2})
            preds_f = ort_sess2.run(None, {'input': blob})

            detections = preds[0][0]
            detections_f = preds_f[0][0]
            end_time = time.perf_counter()
        except Exception as e:
            print(f"Error copying image: {e}")
            return None, None, None

        if debug:
            print('det', end_time - start_time)
        return input_image, detections, detections_f

    # supress overlapping detections which have lower probabilities
    def non_maximum_supression(self, input_image, detections_f, detections, c=0.25, pf=0.6, pp=0.6, str=""):
        start_time = time.perf_counter()
        index_face = cv2.dnn.NMSBoxes(detections_f[:, :4].tolist(), detections_f[:, 4].tolist(), pf, c)
        index_person = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), pp, c)
        end_time = time.perf_counter()
        if debug:
            print('nms: ', end_time - start_time)
        return index_face, index_person
    # Performs following functions
    def extract_text(self, image, bboxes, bboxes_p, tracker, camid):
        start_time = time.perf_counter()
        scale = image.shape[1] / 320
        if len(bboxes_p) < 1: return image
        bboxes_p[:, 2] = bboxes_p[:, 2] - bboxes_p[:, 0]
        bboxes_p[:, 3] = bboxes_p[:, 3] - bboxes_p[:, 1]
        d = bboxes_p[:, :4] * scale
        bboxes_p = np.hstack([d, bboxes_p[:, 4].reshape([-1, 1])])
        bboxes_p = bboxes_p.astype(np.int32)
        if len(bboxes) > 0:
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
            bboxes = bboxes[:, :4] * scale
            bboxes = bboxes.astype(np.int32)

        detections = yolo_detections_to_norfair_detections(bboxes_p, track_points='bbox')
        for detection in detections:
            cut = get_cutout(detection.points, image)
            if cut.shape[0] > 0 and cut.shape[1] > 0:
                detection.embedding = get_hist(cut)
            else:
                detection.embedding = None
        tracked_objects = tracker.update(detections=detections)

        for x in tracked_objects:
            found = False
            for p in self.person_list:
                if len(self.person_list) > 0 and x.id == p.person_id:
                    found = True
                    p.det_person = np.array([*x.last_detection.points[0], *x.last_detection.points[1]])
                    z = p.det_person
                    a = [iouc(xywh2x1y1x2y2(i), z) for i in bboxes]
                    if a:
                        if not all(max(a) == 0):
                            index = np.argmax(a)
                            p.det_face = bboxes[index]
                        else:
                            p.det_face = []

                    if p.firstTime or p.firstTimeName:
                        p.firstTime = False
                        if p.firstTimeName:
                            p.firstTimeNameCheck = False
                            p.firstTimeName = False

                        img = image[p.det_person[1]:p.det_person[3], p.det_person[0]:p.det_person[2]]
                        filename = str(np.random.randint(100000000)) + '.jpg'

                        # ✅ Correct logic: First check if it's unknown (na)
                        if p.person_name == 'na':
                            folder = "D:/New folder/OCR/detected/unclear"
                        else:
                            status = get_person_status(p.person_name, camid)  # 'whitelisted' or 'notwhitelisted'
                            folder = f"D:/New folder/OCR/detected/{status}"

                        os.makedirs(folder, exist_ok=True)
                        file = os.path.join(folder, filename)
                        cv2.imwrite(file, img)

            if not found:
                d = Person(datetime.datetime.now(), datetime.datetime.now(), 0, 3, 'na', x.id)
                d.status = 'An unidentified person has appeared in camera'
                d.det_person = np.array([*x.last_detection.points[0], *x.last_detection.points[1]])

                img = image[d.det_person[1]:d.det_person[3], d.det_person[0]:d.det_person[2]]

                if d.firstTimeNA:
                    d.firstTimeNA = False
                    filename = str(np.random.randint(100000000)) + '.jpg'
                    file = f'D:/New folder/OCR/detected/unclear/{filename}'
                    cv2.imwrite(file, img)

                z = d.det_person
                a = [iouc(xywh2x1y1x2y2(i), z) for i in bboxes]
                if a:
                    index = np.argmax(a)
                    d.det_face = bboxes[index]
                else:
                    d.det_face = []
                self.person_list.append(d)

        for i, p in enumerate(self.person_list):
            found = False
            for x in tracked_objects:
                if len(tracked_objects) > 0:
                    if x.id == p.person_id:
                        found = True
            if not found:
                self.person_list[i].status = 'Person has left the scene'
                self.person_list.remove(p)

        try:
            if len(self.person_list) > 0:
                images = []
                batch = []

                for bb in self.person_list:
                    if len(bb.det_face) > 0:
                        x1, y1, width, height = bb.det_face
                        x2, y2 = x1 + width, y1 + height
                        images.append(image[y1:y2, x1:x2])

                batch = []
                for x in images:
                    img = cv2.resize(x, (112, 112))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))  # CHW
                    img = img.astype(np.float32) / 255.0
                    batch.append(img)

                if len(batch) > 0:
                    batch = np.stack(batch, axis=0)  # shape (N, 3, 112, 112)
                    pred = ort_sess_r.run(None, {'input.1': batch})


                if len(batch) > 0:
                    pred = ort_sess_r.run(None, {'input.1': batch})
                    scores = [{x: compute_sim(p, y[0]) for x, y in users.items()} for p in pred[0]]
                    names = []
                    for emb in scores:
                        if emb:
                            data = {x: y for x, y in emb.items() if y == max(emb.values()) and y > 0.35}
                        else:
                            print("[WARNING] No embeddings available. Skipping recognition.")
                            data = {}
                        if len(data) > 0:
                            names.append([*data][0])
                        else:
                            names.append("na")

                    for bb, n in zip(self.person_list, names):
                        if len(bb.det_face) > 0:
                            if n != 'na':
                                if bb.firstTimeNameCheck:
                                    bb.firstTimeName = True
                                bb.person_name = n

                            now = datetime.datetime.now()
                            elapsed = (now - bb.last_seen).total_seconds()
                            should_create_entry = not bb.db_entry_created or elapsed > MIN_REENTRY_SECONDS

                            if should_create_entry:
                                bb.db_entry_created = True
                                bb.last_seen = now

                                x, y, w, h = bb.det_face
                                color = (255, 0, 0) if bb.person_name != 'na' else (255, 0, 255)
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                cv2.rectangle(image, (x, y - 30), (x + w, y), color, -1)
                                cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)
                                status = get_person_status(bb.person_name, camid)
                                label = f"{bb.person_name} {status}"
                                cv2.putText(image, label, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                                findings = "unclear" if bb.person_name == "na" else status

                                face_img = image[y:y+h, x:x+w]
                                filename = f"{int(time.time())}_{bb.person_id}.jpg"
                                folder_path = os.path.join("D:/New folder/OCR/detected", findings)
                                os.makedirs(folder_path, exist_ok=True)
                                save_path = os.path.join(folder_path, filename)
                                cv2.imwrite(save_path, face_img)

                                print(f"[Saved] {save_path}")
                                rel_path = f"{findings}/{filename}"

                                payload = {
                                    "cam": str(camid),
                                    "filepath": rel_path,
                                    "findings": findings,
                                    "person": {
                                        "name": bb.person_name,
                                        "status": status,
                                        "color": "#ffffff"
                                    },
                                    "seen": 1
                                }

                                try:
                                    res = requests.post("http://localhost:3000/api/detectedframes", json=payload)
                                    print(f"[INFO] Sent detection to DB. Status code: {res.status_code}")
                                except Exception as e:
                                    print(f"[ERROR] Failed to send to DB: {e}")
                            else:
                                x, y, w, h = bb.det_face
                                color = (255, 0, 0) if bb.person_name != 'na' else (255, 0, 255)
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                cv2.rectangle(image, (x, y - 30), (x + w, y), color, -1)
                                cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)
                                status = get_person_status(bb.person_name, camid)
                                label = f"{bb.person_name} {status}"
                                cv2.putText(image, label, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                    # In the loop where person_list is cleaned:
                    current_ids = {x.id for x in tracked_objects}
                    for p in list(self.person_list):
                        if p.person_id not in current_ids:
                            p.db_entry_created = False
                            self.person_list.remove(p)

        except BaseException as e:
            print('exception')
            print(e, e.__traceback__.tb_lineno)

        norfair.draw_tracked_boxes(image, tracked_objects, draw_labels=True)
        end_time = time.perf_counter()
        if debug:
            print('recog: ', end_time - start_time)
        return image

    def yolo_predictions(self, frame, tracker, camid):
        start_time = time.perf_counter()
        frames = frame
        input_image, detections, detections_f = self.get_detections(frames)
        detections = detections[detections[:, 5] >= 0.90]

        indF = cv2.dnn.NMSBoxes(detections_f[:, :4].tolist(), detections_f[:, 4].tolist(), 0.6, 0.4)
        indP = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), 0.40, 0.4)
        if len(indF) > 0:
            indF = xywh2xyxy(detections_f[indF][:, :4])
        else:
            indF = np.array([])
        if len(indP) > 0:
            indP = xywh2xyxy(detections[indP][:, :5])
        else:
            indP = np.array([])

        result_img = self.extract_text(frames, indF, indP, tracker, camid)
        end_time = time.perf_counter()

        if debug:
            print('loop: ', end_time - start_time)
        return result_img


def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(
            snd_embedding, detection_fst.embedding, cv2.HISTCMP_CORREL
        )
        if distance < 0.5:
            return distance
    return 1


def iouc(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=0)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=0)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def xywh2xyxy(x):
    x[:, 0] = x[:, 0] - x[:, 2] / 2
    x[:, 1] = x[:, 1] - x[:, 3] / 2
    x[:, 2] = x[:, 0] + x[:, 2]
    x[:, 3] = x[:, 1] + x[:, 3]
    return x


def xywh2x1y1x2y2(x):
    y = np.copy(x[:4])
    y[2] = y[0] + y[2]
    y[3] = y[1] + y[3]
    return y


def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor, track_points: str = "bbox"
) -> List[Detection]:
    norfair_detections: List[Detection] = []
    for detection_as_xyxy in yolo_detections:
        bbox = np.array(
            [
                [detection_as_xyxy[0], detection_as_xyxy[1]],
                [detection_as_xyxy[0] + detection_as_xyxy[2], detection_as_xyxy[1] + detection_as_xyxy[3]],
            ]
        )
        norfair_detections.append(
            Detection(
                points=bbox,
            )
        )
    return norfair_detections


d = Detections(320, 320, 320, 320)


def getEmbed(file):
    """Generate face embedding from an image file with robust error handling"""
    try:
        # Load image with validation
        img = cv2.imread(file)
        if img is None:
            raise ValueError(f"Could not read image: {file}")
        
        # Get detections with error handling
        input_image, detections, detections_f = d.get_detections(img)
        if detections_f is None or len(detections_f) == 0:
            raise ValueError("No faces detected")
        
        # Process detections safely
        indF, _ = d.non_maximum_supression(input_image, detections_f, np.zeros((0, 6)))
        if len(indF) == 0:
            raise ValueError("No valid faces after NMS")
        
        # Convert and scale bounding box
        boxes_np = xywh2xyxy(detections_f[indF][:, :4])
        scale = input_image.shape[1] / 320
        boxes_np = boxes_np * scale
        
        # Get first face with boundary checks
        x1, y1, x2, y2 = boxes_np[0].astype(np.int32)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        
        # Validate face region
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid face coordinates")
            
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            raise ValueError("Empty face crop")
        
        # Preprocess face
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = np.transpose(face, (2, 0, 1))  # CHW format
        face = (face / 255.0).astype(np.float32)  # Normalize
        
        # Generate embedding
        blob = np.expand_dims(face, axis=0)  # Add batch dimension
        pred = ort_sess_r.run(None, {'input.1': blob})
        return pred[0][0]  # Return first embedding
        
    except Exception as e:
        print(f"Error in getEmbed for {file}: {str(e)}")
        return np.zeros(512)  # Return zero embedding on failure



# Load embeddings from local files
# users = {
#     "imran": getEmbed(file=r"./OCR/data2/imran.PNG"),
#     "hammad": getEmbed(file=r"./OCR/data2/hammad.jpeg"),
#     "Abu": getEmbed(file=r"./OCR/data2/Abu.jpeg"),
#     "Fahad": getEmbed(file=r"./OCR/data2/Fahad.jpeg"),
    
# }

# Load face embeddings from a folder and save as users.pkl
def generate_encodings_from_folder(WHITELIST_FOLDER, USERS_PKL_PATH):
    users = {}
    image_paths = glob.glob(os.path.join(WHITELIST_FOLDER, "*.*"))
    for path in image_paths:
        try:
            name = os.path.splitext(os.path.basename(path))[0]
            users[name] = getEmbed(path)
            print(f"[INFO] Encoded: {name}")
        except Exception as e:
            print(f"[ERROR] Failed to encode {path}: {e}")
    
    os.makedirs(os.path.dirname(USERS_PKL_PATH), exist_ok=True)

    with open(USERS_PKL_PATH, "wb") as f:
        pickle.dump(users, f)
    print(f"[INFO] Saved all encodings to {USERS_PKL_PATH}")


# Load users from pickle or generate them if missing
def load_or_generate_users(pkl_path="OCR/users.pkl", folder_path="OCR/whitelisted"):
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            users = pickle.load(f)
        print("[INFO] Loaded users from pickle.")
    else:
        print("[INFO] Pickle not found. Generating from folder...")
        generate_encodings_from_folder(folder_path, pkl_path)
        with open(pkl_path, "rb") as f:
            users = pickle.load(f)
    return users

# Example use
users = load_or_generate_users(USERS_PKL_PATH, WHITELIST_FOLDER)

def compute_sim(feat1, feat2):
    if np.all(feat1 == 0) or np.all(feat2 == 0):
        return 0.0  # No similarity for empty embeddings
    return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))


def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()