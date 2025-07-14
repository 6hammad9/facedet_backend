import numpy as np
from numba import njit
import logging


@njit()
def nmx(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


@njit(fastmath=True)
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # x[:, 2] = x[:, 2]  # bottom right x
    # x[:, 3] = x[:, 3]  # bottom right y
    x[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    x[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return x


# @njit(fastmath=True)
def nms(dets, threshold, nms_threshold):
    dets = dets[dets[:, 4] >= threshold]# & np.where(dets[:, 5] >= nms_threshold)[0]
    order = np.where(dets[:, 5] >= threshold)[0]

    dets = dets[order, :]
    pre_det = dets[:, 0:6]
    # lmks = dets[:, 5:15]
    pre_det = xywh2xyxy(pre_det)
    keep = nmx(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    if not len(keep): return dets
    det_out = pre_det[keep, :]
    # lmks = lmks[keep, :]
    # lmks = lmks.reshape((lmks.shape[0], -1, 2))
    return det_out  # , lmks
