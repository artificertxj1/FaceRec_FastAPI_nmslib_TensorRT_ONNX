import time
import logging
from typing import Union 

import numpy as np


from FaceModels.backends.trt_backend import CenterfaceTRT as ctr
from FaceModels.backends.onnx_backend import CenterfaceONNX as con

def nms(dets, thresh = 0.4):
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

class Centerface(object):
    def __init__(self, inference_backend:Union[ctr, con], landmarks=True):
        self.landmarks = landmarks
        self.net = inference_backend
        self.nms_threshold = 0.3
        self.input_shape = (1, 3, 480, 640)
        
    def __call__(self, img, threshold=0.5):
        return self.detect(img, threshold)
    
    def prepare(self, nms: float = 0.3):
        self.nms_threshold = nms
        self.net.prepare()
        self.input_shape = self.net.input_shape

    def detect(self, img: np.ndarray, threshold: float = 0.4):
        h, w = img.shape[:2]
        blob = np.expand_dims(img[:, :, (2, 1, 0)].transpose(2, 0, 1), axis=0).astype("float32")
        t0 = time.time()
        heatmap, scale, offset, lms = self.net.run(blob)
        t1 = time.time()
        logging.debug(f"Centerface inference took: {t1 - t0}")
        #print(f"Centerface inference took: {t1 - t0}")
        return self.postprocess(heatmap, lms, offset, scale, (h, w), threshold)

    def postprocess(self, heatmap, lms, offset, scale, size, threshold):
        t0 = time.time()
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, size, threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, size, threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2], dets[:, 1:4:2]
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2], lms[:, 1:10:2]
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        t1 = time.time()
        logging.debug(f"Centerface postprocess took: {t1 - t0}")
        boxes, probs = dets[:, 0:4], dets[:, 4]
        if self.landmarks:
            return boxes, probs, lms
        else:
            return boxes, probs, None

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = nms(boxes, self.nms_threshold)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
                lms = lms.reshape((-1, 5, 2))
        if self.landmarks:
            return boxes, lms
        else:
            return boxes
        
    
