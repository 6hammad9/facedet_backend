import datetime
import cv2
import jsonpickle
import os
import pickle
from numpy.linalg import norm
import os
import warnings
from line_profiler_pycharm import profile
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

from flask_mysqldb import MySQL
import mysql.connector
BASE_PATH = os.getcwd()

# IMPORTANT
# we use OnnxRuntime for inference purposes, not pytorch, tensorflow or opencv. it is the fastest solution out there
# only one execution provider can be used at a time. for now we are using Cuda execution provider
# TensorRT is fastest and recommended, it takes 5 extra mins to run the project with tensorRT but overall performance
# improves by 50% or more
# to use TensorRT, comment ['CUDAExecutionProvider'] line # 39, and uncomment line 34 to 38, same for other ORT sessions

# this model is used to detect face, feel free to use ur own models from the net, but they must be converted to onnx @16bit
ort_sess2 = ort.InferenceSession('./OCR/models/f2.onnx',
                                 providers=
                                 # ['TensorrtExecutionProvider'], options=({
        # 'trt_fp16_enable': True,
        # 'trt_dla_enable': True
    #
    # })
                                 ['CUDAExecutionProvider'],
                                 # ['CPUExecutionProvider'],
                                 )

# this model is used to detect Person, feel free to use ur own models from the net, but they must be converted to onnx @16bit
# reason we are detecting person is for it to be used in tracking module. tracking directly the faces doesnt work.
ort_sess = ort.InferenceSession('./OCR/models/yolov5m_dynamic.onnx',
                                providers=
                                # ['TensorrtExecutionProvider'], options=({
        # 'trt_fp16_enable': True,
        # 'trt_dla_enable': True
    #
    # })
                                ['CUDAExecutionProvider'],
                                # ['CPUExecutionProvider'],
                                )
# this model is used to recognise face, feel free to use ur own models from the net, but they must be converted to onnx @16bit
# look for Arcface on net, their other models can be use here
ort_sess_r = ort.InferenceSession(r'./OCR/models/glintr100.onnx',
                                  providers=
                                  # ['TensorrtExecutionProvider'], options = ({
        # 'trt_fp16_enable': True,
        # 'trt_dla_enable': True
    #
    # })
                                  ['CUDAExecutionProvider'],
                                  # ['CPUExecutionProvider'],
                                  )

# simple Person object with details we are interested in


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

# Detection class instantiated in app.py
class Detections:
    def __init__(self, wf, hf, wp, hp):
        self.WIDTH_FACE = wf
        self.HEIGHT_FACE = hf
        self.WIDTH_PERSON = wp
        self.HEIGHT_PERSON = hp
        self.person_list = []

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
    @profile
    def non_maximum_supression(self, input_image, detections_f, detections, c=0.25, pf=0.6, pp=0.6, str=""):
        start_time = time.perf_counter()

        index_face = cv2.dnn.NMSBoxes(detections_f[:, :4].tolist(), detections_f[:, 4].tolist(), pf, c)
        index_person = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), pp, c)
        end_time = time.perf_counter()
        if debug:
            print('nms: ', end_time - start_time)
        return index_face, index_person
    def extract_text(self, image, bboxes, bboxes_p, tracker, camid, cursor = None):
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
                        # Crop the face
                        # x1, y1, width, height = bb.det_face
                        # x2, y2 = x1 + width, y1 + height


                        # # Ensure the bounding box coordinates are within the image dimensions
                        # x1 = max(0, x1)
                        # y1 = max(0, y1)
                        # x2 = min(image.shape[1], x2)
                        # y2 = min(image.shape[0], y2)
                        #
                        # # Crop the image only if the bounding box coordinates are valid
                        # if x2 > x1 and y2 > y1:
                        #     cropped_face = image[y1:y2, x1:x2]
                        #     images.append(cropped_face)


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
                if len(batch) > 0:
                    pred = ort_sess_r.run(None, {'input.1': batch})

                    embeddings_file = "newpk1.pkl"
                    with open(embeddings_file, "rb") as f:
                        known_embeddings = pickle.load(f)

                    scores = [{i: compute_sim(p, y['embedding'][0]) for i, y in known_embeddings.items()} for p in pred[0]]
                    names = []
                    findings_list = []
                    for emb in scores:
                        data = {x: y for x, y in emb.items() if y == max(emb.values()) and y > 0.35}
                        max_similarity = max(emb.values())
                        print(
                            f"Max similarity (confidence): {max_similarity}")
                        if len(data) > 0:
                            index = [*data][0]
                            name = known_embeddings[index]['name']
                            cam_id = known_embeddings[index]['cam_id']

                            if cam_id == "cam"+camid:
                                names.append(name)
                                findings_list.append('whitelisted')
                            else:
                                names.append(f"{name}(nw)")
                                findings_list.append('notwhitelisted')
                        elif max_similarity < 0.05:
                              names.append("unclear")
                              findings_list.append('unclearpic')

                        for i, p in enumerate(self.person_list):
                            if len(names) > i and len(names[i]) > 2:
                                person_name = names[i]
                                findings = findings_list[i]
                                if person_name != "unclear":
                                    if p.firstTime or p.firstTimeName:
                                        p.firstTime = False
                                        if p.firstTimeName:
                                            p.firstTimeNameCheck = False
                                            p.firstTimeName = False

                                        connection = mysql.connector.connect(host='localhost', database='emacsdb',
                                                                             user='root', password='')
                                        cursor = connection.cursor()
                                        getdatetime = datetime.datetime.now().isoformat()
                                        setcamid = "cam" + camid
                                        filename = str(np.random.randint(100000000)) + '.jpg'

                                        sql = "INSERT INTO detected_frames (cam, filepath, findings, personname, datetime) VALUES (%s, %s, %s, %s, %s)"
                                        val = (setcamid, filename, findings, person_name, getdatetime)
                                        if findings == 'whitelisted':
                                            file = f'D:\\xampp\\htdocs\\emacsPanel\\detected_pictures\\whitelisted\\{filename}'
                                            try:
                                                cv2.imwrite(file, b_img)
                                            except Exception as e:
                                                raise Exception(f"Could not write image: {e}")

                                        elif findings == 'notwhitelisted':
                                            file = f'D:\\xampp\\htdocs\\emacsPanel\\detected_pictures\\notwhitelisted\\{filename}'
                                            try:
                                                cv2.imwrite(file, b_img)
                                            except Exception as e:
                                                raise Exception(f"Could not write image: {e}")
                                        elif findings == 'unclear':
                                            file = f'D:\\xampp\\htdocs\\emacsPanel\\detected_pictures\\unclearpic\\{filename}'
                                            try:
                                                cv2.imwrite(file, b_img)
                                            except Exception as e:
                                                raise Exception(f"Could not write image: {e}")

                                        cursor.execute(sql, val)
                                        connection.commit()
                                        cursor.close()

                                        if connection:
                                            connection.close()

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
        self.output_path = output_path if output_path else os.getcwd()
        self.embeddings_file = os.path.join(self.output_path, "newpk1.pkl")

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
        connection = mysql.connector.connect(host='localhost',
                                             database='emacsdb',
                                             user='root',
                                             password='')
        cursor = connection.cursor()
        sql_select_query = """SELECT * FROM person_info"""
        cursor.execute(sql_select_query)
        rv = cursor.fetchall()

        embeddings = self.load_embeddings()

        for row in rv:
            id = row[0]
            name = row[1]
            cam_id = row[2]

            print(name, id)
            sql_select_query = """SELECT * FROM whitelisted_pictures WHERE person_id=%s limit 1"""
            input_data = (id,)
            cursor.execute(sql_select_query, input_data)
            rv = cursor.fetchall()
            for row in rv:
                filepath = row[2]
                embededFile = getEmbed(
                    file="D:/xampp/htdocs/emacsPanel/detected_pictures/uploaded/" + filepath.replace('\\', '/'))
                index = len(embeddings)
                embeddings[index] = {'name': name, 'cam_id': cam_id, 'embedding': embededFile}
                print(embeddings)

        cursor.close()
        if connection:
            connection.close()

        self.save_embeddings(embeddings)


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
output_path = "E:\pc2"
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
