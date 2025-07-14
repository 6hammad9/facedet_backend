import datetime
import cv2
import jsonpickle
import os
import pickle
from numpy.linalg import norm
import warnings
from pymongo import MongoClient
from bson.binary import Binary
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
import onnxruntime as ort

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ONNX Runtime configuration
def get_onnx_providers():
    try:
        ort.get_device()
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    except:
        return ['CPUExecutionProvider']

# Initialize ONNX sessions
ort_sess2 = ort.InferenceSession(os.path.join(BASE_PATH, 'models/f2.onnx'),
                                providers=get_onnx_providers())

ort_sess = ort.InferenceSession(os.path.join(BASE_PATH, 'models/yolov5m_dynamic.onnx'),
                               providers=get_onnx_providers())

ort_sess_r = ort.InferenceSession(os.path.join(BASE_PATH, 'models/glintr100.onnx'),
                                 providers=get_onnx_providers())

# MongoDB Connection Class with your specific collections
class mongo:
    def __init__(self):
        try:
            self.client = MongoClient(
                'mongodb://localhost:3000/',
                serverSelectionTimeoutMS=5000
            )
            self.client.server_info()  # Test connection
            self.db = self.client['EMACS']
            
            # Initialize all your collections with exact names
            self.camera_info = self.db['CameraInfo']
            self.detected_frames = self.db['DetectedFrames'] 
            self.person_info = self.db['PersonInfo']
            self.whitelisted_pictures = self.db['WhitelistedPictures']
            self.department = self.db['Department']
            
            print("Successfully connected to MongoDB on port 3000")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise

    # Add these methods to interact with your collections
    def get_all_persons(self):
        """Get all persons from PersonInfo collection"""
        return list(self.person_info.find({}))

    def get_whitelisted_picture(self, person_id):
        """Get whitelisted picture for a person"""
        return self.whitelisted_pictures.find_one({"person_id": person_id})

    def save_detection(self, detection_data):
        """Save to DetectedFrames collection"""
        try:
            if 'image' in detection_data:
                detection_data['image'] = Binary(detection_data['image'])
            return self.detected_frames.insert_one(detection_data)
        except Exception as e:
            print(f"Error saving detection: {e}")
            return None

    def update_face_embedding(self, person_id, name, cam_id, embedding):
        """Update face embedding in PersonInfo"""
        try:
            self.person_info.update_one(
                {'_id': person_id},
                {'$set': {
                    'name': name,
                    'cam_id': cam_id,
                    'embedding': Binary(pickle.dumps(embedding)),
                    'last_updated': datetime.datetime.now()
                }},
                upsert=True
            )
        except Exception as e:
            print(f"Error updating face embedding: {e}")
class Person:
    def __init__(self, det_time, exit_time, duration, det_history, person_name,
                 person_id, status='', f=None, p=None):
        if p is None:
            p = list()
        if f is None:
            f = list()
        self.det_time = det_time
        self.firstTime = True
        self.firstTimeName = True
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

    def details(self):
        return 3

debug = False

class Detections:
    def __init__(self, wf, hf, wp, hp):
        self.WIDTH_FACE = wf
        self.HEIGHT_FACE = hf
        self.WIDTH_PERSON = wp
        self.HEIGHT_PERSON = hp
        self.person_list = []

    # Keep existing detection methods unchanged
    def get_detections(self, img):
        # Existing implementation remains the same
        pass

    
    def get_detections(self, img):
        start_time = time.perf_counter()
        try:
            image = img.copy()
            row, col, d = image.shape
            # add black regions in image to preserve aspect ratio
            max_rc = max(row, col)
            input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
            input_image[0:row, 0:col] = image

            # converts the image to standard format that can be used to run inference, this includes, adding extra dimension,
            # swapping 1st and 3rd channel, normalizing the pixel values
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
        image = img.copy()
        row, col, d = image.shape
        # add black regions in image to preserve aspect ratio
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # converts the image to standard format that can be used to run inference, this includes, adding extra dimension,
        # swapping 1st and 3rd channel, normalizing the pixel values
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.WIDTH_FACE, self.HEIGHT_FACE), swapRB=True,
                                     crop=False)
        blob2 = blob

        # run inference on the processed image both for face n person
        preds = ort_sess.run(None, {'images': blob2})
        preds_f = ort_sess2.run(None, {'input': blob})

        detections = preds[0][0]
        detections_f = preds_f[0][0]
        end_time = time.perf_counter()

        if debug:
            print('det', end_time - start_time)
        # return all the detections of face and person for further processing
        return input_image, detections, detections_f
    
    def non_maximum_supression(self, input_image, detections_f, detections, c=0.25, pf=0.6, pp=0.6, str=""):
        start_time = time.perf_counter()

        index_face = cv2.dnn.NMSBoxes(detections_f[:, :4].tolist(), detections_f[:, 4].tolist(), pf, c)
        index_person = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), pp, c)
        end_time = time.perf_counter()
        if debug:
            print('nms: ', end_time - start_time)
        return index_face, index_person
    def extract_text(self, image, bboxes, bboxes_p, tracker, camid):
        start_time = time.perf_counter()
        scale = image.shape[1] / 320
        if len(bboxes_p) < 1: return image
        
        # Existing processing logic remains the same until detection handling
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
        detections = yolo_detections_to_norfair_detections(
            bboxes_p, track_points='bbox'
        )
        for detection in detections:
            cut = get_cutout(detection.points, image)
            if cut.shape[0] > 0 and cut.shape[1] > 0:
                detection.embedding = get_hist(cut)
            else:
                detection.embedding = None
        tracked_objects = tracker.update(detections=detections)
        if tracked_objects:
            pass
        for x in tracked_objects:
            found = False
            for p in self.person_list:
                if len(self.person_list) > 0 and x.id == p.person_id:
                    if x.id == p.person_id:
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
                        else:
                            p.det_face = []
                        b_img = image[p.det_person[1]:p.det_person[3], p.det_person[0]:p.det_person[2]]

            if not found:
                d = Person(datetime.datetime.now(), datetime.datetime.now(), 0, 3, 'na', x.id)
                d.status = 'An unidentified person has appeared in camera'
                d.det_person = np.array([*x.last_detection.points[0], *x.last_detection.points[1]])
                b_img = image[d.det_person[1]:d.det_person[3],d.det_person[0]:d.det_person[2]]
                if d.firstTimeNA:
                    d.firstTimeNA =False
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

                for x in images:
                    resized_image = x.copy()
                    img = cv2.resize(x, (112, 112))

                    # Save the resized image before reshaping and normalization


                    img = img.reshape(1, 112, 112, 3)
                    samples = np.asarray(img, 'float32')
                    blob = cv2.dnn.blobFromImage(samples[0], 1 / 255, (112, 112), swapRB=True)
                    batch = np.append(batch, blob, axis=0) if len(batch) > 0 else np.array(blob)
                # Existing image processing logic
                
                if len(batch) > 0:
                    pred = ort_sess_r.run(None, {'input.1': batch})
                    # embeddings_file = "newpk1.pkl"
                    # Load embeddings from MongoDB
                    known_embeddings = {}
                    for doc in mongo.known_faces.find():
                        known_embeddings[str(doc['_id'])] = {
                            'name': doc['name'],
                            'cam_id': doc['cam_id'],
                            'embedding': pickle.loads(doc['embedding'])
                        }

                    scores = [{i: compute_sim(p, y['embedding']) for i, y in known_embeddings.items()} 
                             for p in pred[0]]
                    
                    names = []
                    findings_list = []
                    for emb in scores:
                        data = {x: y for x, y in emb.items() if y == max(emb.values()) and y > 0.35}
                        max_similarity = max(emb.values(), default=0)
                        
                        if len(data) > 0:
                            index = list(data.keys())[0]
                            name = known_embeddings[index]['name']
                            cam_id = known_embeddings[index]['cam_id']
                            
                            # MongoDB detection saving
                            if cam_id == f"cam{camid}":
                                names.append(name)
                                findings_list.append('whitelisted')
                            else:
                                names.append(f"{name}(nw)")
                                findings_list.append('notwhitelisted')
                                
                            # Save to MongoDB
                            if any(p.firstTime for p in self.person_list):
                                _, img_encoded = cv2.imencode('.jpg', b_img)
                                detection_data = {
                                    'cam_id': f"cam{camid}",
                                    'timestamp': datetime.datetime.now(),
                                    'person_name': name,
                                    'findings': findings_list[-1],
                                    'image': img_encoded.tobytes(),
                                    'metadata': {
                                        'detection_box': p.det_face.tolist(),
                                        'tracking_id': p.person_id
                                    }
                                }
                                mongo.save_detection(detection_data)
                                
                        elif max_similarity < 0.05:
                            names.append("unclear")
                            findings_list.append('unclearpic')
                            # Save unclear detection
                            _, img_encoded = cv2.imencode('.jpg', b_img)
                            detection_data = {
                                'cam_id': f"cam{camid}",
                                'timestamp': datetime.datetime.now(),
                                'findings': 'unclearpic',
                                'image': img_encoded.tobytes(),
                                'metadata': {
                                    'detection_box': p.det_face.tolist(),
                                    'tracking_id': p.person_id
                                }
                            }
                            mongo.save_detection(detection_data)
                    for bb, n in zip(self.person_list, names):
                        if len(bb.det_face) > 0:
                            if n != 'na':
                                if bb.firstTimeNameCheck:
                                    bb.firstTimeName = True
                                bb.person_name = n
                            x, y, w, h = bb.det_face
                            if bb.person_name != 'na':

                                color = (255, 0, 0)
                            else:
                                color = (255, 0, 255)
                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                            cv2.rectangle(image, (x, y - 30), (x + w, y), color, -1)
                            cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)
                            cv2.putText(image, bb.person_name, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255), 2)
            else:
                pass   
                    # Rest of your drawing/annotation logic remains the same

        except BaseException as e:
            print('exception')
            print(e, e.__traceback__.tb_lineno)

        norfair.draw_tracked_boxes(image, tracked_objects, draw_labels=True)
        end_time = time.perf_counter()
        if debug:
            print('recog: ', end_time - start_time)
        return image

    def yolo_predictions(self, frame, tracker,camid, cursor = None):
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

        result_img = self.extract_text(frames, indF, indP, tracker, camid, cursor)
        end_time = time.perf_counter()

        if debug:
            print('loop: ', end_time - start_time)
        return result_img

class EmbeddingHandler:
    def __init__(self, output_path=None):
        self.output_path = os.path.normpath(output_path) if output_path else BASE_PATH
        os.makedirs(self.output_path, exist_ok=True)
    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = {}
        return embeddings

    def save_embeddings(self, embeddings):
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
    def update_embeddings(self):
        if not mongo:
            print("MongoDB not available")
            return

        try:
            # Get persons from PersonInfo collection
            persons = mongo.get_all_persons()
            
            for person in persons:
                person_id = person['_id']  # Use the _id field directly
                name = person.get('name', 'unknown')
                cam_id = person.get('cam_id', '')

                # Get image from WhitelistedPictures
                wp = mongo.get_whitelisted_picture(person_id)
                if wp and 'image' in wp:
                    try:
                        # Process the image
                        img_data = wp['image']
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Save temp file
                        temp_path = os.path.join(self.output_path, f"temp_{person_id}.jpg")
                        cv2.imwrite(temp_path, img)
                        
                        # Get embedding
                        embedding = getEmbed(file=temp_path)
                        os.remove(temp_path)
                        
                        # Update in PersonInfo
                        mongo.update_face_embedding(person_id, name, cam_id, embedding)
                        
                    except Exception as e:
                        print(f"Error processing image for {name}: {str(e)}")

            print("Embeddings updated successfully in MongoDB")
        except Exception as e:
            print(f"Error updating embeddings: {e}")

# Rest of the helper functions remain unchanged
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
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    x[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    x[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return x
def xywh2x1y1x2y2(x):
    y = np.copy(x[:4])
    y[2] = y[0] + y[2]  # bottom right x
    y[3] = y[1] + y[3]  # bottom right y
    return y
def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor, track_points: str = "bbox"
) -> List[Detection]:
    norfair_detections: List[Detection] = []
    for detection_as_xyxy in yolo_detections:
        bbox = np.array(
            [
                [detection_as_xyxy[0], detection_as_xyxy[1]],
                [detection_as_xyxy[0] + detection_as_xyxy[2]
                    , detection_as_xyxy[1] + detection_as_xyxy[3]],
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
    img = cv2.imread(file)
    i, _, detections_f = d.get_detections(img)
    indF, indP = d.non_maximum_supression(i, detections_f, _)
    boxes_np = xywh2xyxy(detections_f[indF][:, :4])
    boxes_np[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
    boxes_np[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
    scale = i.shape[1] / 320
    boxes_np = boxes_np * scale
    image = boxes_np[0].astype(np.int32)
    print(file)
    x1, y1, width, height = image
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    images = cv2.resize(face, (112, 112))
    images = images.reshape(1, 112, 112, 3)
    # images = np.transpose(images, (0, 3, 1, 2))
    samples = asarray(images, 'float32')
    # TODO, 1/127.5 for std and mean(127.5,,)
    blob = cv2.dnn.blobFromImage(samples[0], 1.0 / 255, (112, 112), swapRB=True)

    pred = ort_sess_r.run(None, {'input.1': blob})
    # pred = model(samples)
    return pred[0]
# pickle_file = "embeddings.pkl"
# embedding_handler = EmbeddingHandler(pickle_file)
users = {}

# pickle_file = "embeddings.pkl"
# embedding_handler = EmbeddingHandler(pickle_file)
# users = embedding_handler.load_embeddings()

# embedding_handler.save_embeddings(users)
def compute_sim(feat1, feat2):
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim
def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


# # Usage example
# # pickle_file = "embeddings.pkl"
# # embedding_handler = EmbeddingHandler(pickle_file)
# pickle_file ="hammad.pkl"
# embedding_handler1 = EmbeddingHandler(pickle_file, person_ids=[14])
# embedding_handler1.update_embeddings()
# # Load embeddings from the pickle file
# loaded_embeddings = embedding_handler1.load_embeddings()
output_path = "D:\\EMACS REACT\\emacs_backend\\uploads"
embedding_handler1 = EmbeddingHandler( output_path=output_path)
embedding_handler1.update_embeddings()
#
# embedding_handler2 = EmbeddingHandler(camid=4, person_ids=[19], output_path=output_path)
# embedding_handler2.update_embeddings()
loaded_embeddings= " "
loaded_embeddings = embedding_handler1.load_embeddings()
# embedding_handler = EmbeddingHandler(None)
# embedding_handler.update_embeddings()
# Print the loaded embeddings
# print("Loaded embeddings from the pickle file:")
# for name, embedding in loaded_embeddings.items():
#     print(f"{name}: {embedding}")
# if __name__ == "__main__":
#     output_path = "E:\pc2"
#     embedding_handler = EmbeddingHandler(output_path=output_path)
#     embedding_handler.update_embeddings()