import time

import onnxruntime as ort
import numpy as np
from numba import njit
import cv2
import asyncio
from line_profiler_pycharm import profile
import onnxruntime as ort
# import matplotlib.pyplot as pl;pl.imshow(image[y1:y2, x1:x2]);pl.show()

image = cv2.imread(r'D:\New folder\OCR\data2\farhan.JPG')
INPUT_WIDTHP = 320
INPUT_HEIGHTP = 320
INPUT_WIDTHF = 320
INPUT_HEIGHTF = 320
#%%
sessionF = ort.InferenceSession(r'D:\New folder\OCR\models\f2.onnx', providers=['CUDAExecutionProvider'])
sessionP = ort.InferenceSession(r'D:\New folder\OCR\models\yolov5m_320.onnx', providers=['CUDAExecutionProvider'])

#%%

#  (1, 3, 320, 320)
def dosmth():
    # image = cv2.imread(r'E:\pytorch-flask-api\OCR\data2\alamzeb.JPG')
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blobF = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTHF, INPUT_HEIGHTF), swapRB=True, crop=False)
    blobP = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTHP, INPUT_HEIGHTP), swapRB=True, crop=False)
    blobP = blobP.astype(np.float16)
    # blobP = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTHP, INPUT_HEIGHTP), swapRB=True, crop=False)
    preds_f = sessionF.run(['output'], {'input': blobF})[0]
    preds = sessionP.run(['output0'], {'images': blobP})[0]


#%%

@njit()
def prepare_image(img):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)

    return img
#%%
%prun dosmth()
#%%
%%timeit -r 100 -n 15
dosmth()
#%%
# %lprun -u 1e-3 -f dosmth dosmth()
%lprun -u 1e-3 -f non_maximum_supression non_maximum_supression(image,preds[0],preds_f[0])

# %%
async def do_person(p):
    return sessionP.run(['output0'], {'images': p})

async def do_face(f):
    return sessionF.run(['output'], {'input': f})

async def person(p):
    # await asyncio.sleep(0)
    print('executing person')
    ss=await do_person(p)
    asyncio
    print('done person')
    # task=asyncio.create_task(sessionP.run(['output0'], {'images': p}))
    return ss

async def face(f):
    # await asyncio.sleep(0)
    # task=asyncio.create_task(sessionF.run(['output'], {'input': f}))
    print('executing face')
    # await asyncio.sleep(0.00001)
    ss=await do_face(f)
    print('done face')
    return ss

async def final(f,p):
    # taskp = asyncio.create_task(person(p))
    # taskf = asyncio.create_task(face(f))
    # return await asyncio.gather( taskp,taskf)
    return await asyncio.gather( face(f),person(p))

#%% async
def dosmth():
    # image = cv2.imread(r'E:\pytorch-flask-api\OCR\data2\alamzeb.JPG')
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blobF = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTHF, INPUT_HEIGHTF), swapRB=True, crop=False)
    blobP = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTHP, INPUT_HEIGHTP), swapRB=True, crop=False)
    blobP = blobP.astype(np.float16)
    res = asyncio.run(final(blobF,blobP))
    preds_f = res[0]
    preds = res[1]
dosmth()

#%%


def get_embedding(self, face_img):
    if not isinstance(face_img, list):
        face_img = [face_img]

    face_img = np.stack(face_img)

    input_size = tuple(face_img[0].shape[0:2][::-1])
    blob = cv2.dnn.blobFromImages(face_img, 1.0 / 1, input_size,
                                  (0,0,0), swapRB=True)

    net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: blob})
    return net_out[0]



#%%


q = asyncio.Queue(maxsize=2000)

async def get_data (data):
    while True:
        await asyncio.sleep(.1)
        await q.put(data[idx,:])
        idx += 1
        #Each row of data is read every second.

async def run_algorithm ():
    while True:
        data_read = await q.get()
        #I do something here with the read data














