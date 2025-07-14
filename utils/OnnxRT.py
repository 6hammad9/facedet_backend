from numba import jit


@jit(nopython=True)
def exec():
    model(image)


# %%

from line_profiler_pycharm import profile
import onnx
import cv2
import numpy as np

path = r"./OCR/models/f2.onnx"
path = r"./OCR/models/yolov5m.onnx"
path = r"./OCR/models/yolov5m_o11.onnx"
path = r"./OCR/models/yolov5m_320.onnx"
path = r"./OCR/models/yolov5m_320_fp16.onnx"
# path = r"./OCR/models/yolov5m_dynamic.onnx"

onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)

# %%
# CV2 DNN
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
# INPUT_WIDTH = 640
# INPUT_HEIGHT = 640

net = cv2.dnn.readNetFromONNX(r"./OCR/models/f2.onnx")

img = cv2.imread(r'E:\pytorch-flask-api\OCR\data2\alamzeb.JPG')

row, col, d = img.shape
max_rc = max(row, col)
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
input_image[0:row, 0:col] = img

# blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
blob = blob.astype(np.float16)
# net.setInput(blob)
# preds = net.forward()
# detections = preds[0]
# 40.5 ms ± 614 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


# %%
# PYTORCH
from onnx2torch import convert
import onnx

onnx_model = onnx.load(r"OCR/models/f2.onnx")
onnx.checker.check_model(onnx_model)
torch_model_1 = convert(onnx_model)
image = cv2.imread('OCR/data2/Haris.jpg').astype(np.float32)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (320, 320))
image = image.astype("float32") / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, 0)

labels = ['Person', 'Face']
torch_model_1.eval()
image = torch.from_numpy(image)
image = image.to(DEVICE)
model = torch_model_1.to(DEVICE)
logits = model(image)

# 48.6 ms ± 2.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


# %%

# ORT CPU,  why was it using gpu?
import onnxruntime as ort

ort_sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

outputs = ort_sess.run(None, {'images': blob})[0]

# 5.08 ms ± 188 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# %%


import onnxruntime as ort

ort_sess = ort.InferenceSession(path,
                                providers=
                                ['CUDAExecutionProvider'],
                                )


outputs = ort_sess.run(None, {'images': blob.astype(np.float16)})[0]

# %%

import onnxruntime as ort

# sess_options = ort.SessionOptions();sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort_sess = ort.InferenceSession(path,
                                providers=
                                ['TensorrtExecutionProvider'],
                                # ['CUDAExecutionProvider'],
                                # ['CPUExecutionProvider'],
                                provider_options=[{
                                    # 'trt_dla_enable': True,
                                    # 'trt_fp16_enable': True,
                                    # 'trt_engine_cache_enable': True,
                                    # 'trt_engine_cache_path': './OCR/models/',
                                    'trt_engine_cache_path': './OCR/models',

                                }],
                                # sess_options=sess_options
                                )

outputs = ort_sess.run(None, {'images': blob})[0]

#%%


# 48.6 ms ± 2.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)      CPU GPU
# 5.08 ms ± 188 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)     ORT
# 1.77 ms ± 18.8 µs per loop(mean ± std. dev. of 7 runs, 100 loops each)    TRT

# 17.1 ms ± 226 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)    O_11
# 23.3 ms ± 444 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)   T_1.10.0
# 17.8 ms ± 380 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)    T_1.11.0
# 17.5 ms ± 282 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)    pdev_12
# 17.9 ms ± 1.01 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)    pdev_O11
# 17.4 ms ± 267 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)    pdev_O13
# 17.5 ms ± 150 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)    pdev_O15
# 8.12 ms ± 1.02 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)  dynamic 320
# 5.97 ms ± 1.06 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)  320
# 4.07 ms ± 392 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)  320 16-bit
# 5.83 ms ± 306 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)  320 dla ON
# 5.72 ms ± 349 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)  sess_options = ort.SessionOptions();sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


# todo, optimization,
# half precision                D
# 8bit quantization             F fallback to model conversion
# dynamic batch                 D
# run TRT with bigger model
# onnx graph simplification     D

# 107.1 ms ± 133 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)           #CPU
# 25.4 ms ± 436 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)           #CUDA
# 17.4 ms ± 436 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)           #TRT
#%%
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import preprocess
preprocess()

model_fp32 = path
model_quant = './quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant)

#%%

%%timeit
ort_sess.run(None, {'images': blob})

# %%
%%timeit -r 15 -n 200
ort_sess.run(None, {'images': blob})
# 4.09 ms ± 308 µs per loop (mean ± std. dev. of 15 runs, 200 loops each)
# 4.26 ms ± 1.09 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)
# 4.25 ms ± 1.08 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)
# 4.25 ms ± 1.08 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)
# 4.32 ms ± 1.06 ms per loop (mean ± std. dev. of 15 runs, 200 loops each)
# %%

# simplify creates issues
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(path)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

#%%

def fun():
    return sum([3,4,5])

#%%
%lprun -f fun fun()



#%%
import asyncio

async def one():
    await asyncio.sleep(2)
    print(1)
    return 1
async def tas():
    res = await asyncio.create_task(printing())
    return res

async def two():
    await asyncio.sleep(2)
    print(2)
    return 2

async def printing1():
    await asyncio.sleep(0)
    [print(x) for x in range(1000000)]
    # return 3
async def printing2():
    await asyncio.sleep(0)

    return 'sadf'
    # return 3

async def gather():
    task = asyncio.create_task(printing2())
    res = await asyncio.gather(one(),two())
    return res

def doit():
    res = asyncio.run(gather())
    return res

#%%
async def asyn():
    task = asyncio.create_task(printing2())
    # task1 = asyncio.create_task(printing1())
    return await asyncio.gather(two(),one(),task,two())
asyncio.run(asyn())
#%%
doit()
asyncio.run(tas())


13468626 129
4420208 434


#%%




















