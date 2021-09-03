import os
import sys
import time
import collections
from typing import Dict, List, Optional, Union
import logging
import asyncio
import pickle
import nmslib

import numpy as np
from numpy.linalg import norm
import cv2

from FaceModels.utils.helpers import to_chunks
from FaceModels.utils import face_align
from FaceModels.utils.helpers import to_chunks
from FaceModels.backends.onnx_backend import CenterfaceONNX, RecONNX
from FaceModels.backends.trt_backend import CenterfaceTRT, RecTRT
from FaceModels.detector.centerface import Centerface




Face = collections.namedtuple("Face", ["bbox", "name", "facedata"])
Face.__new__.__defaults__ = (None,) * len(Face._fields)




def get_onnx_detector(onnx_file_path:str, nms_threshold:float=0.35, landmarks:bool=True):
    backend = CenterfaceONNX(onnx_file_path)
    engine  = Centerface(backend, landmarks=landmarks)
    engine.prepare(nms=nms_threshold)
    return engine

def get_onnx_recognitor(onnx_file_path:str):
    engine = RecONNX(onnx_file_path)
    engine.prepare()
    return engine 

def get_trt_detector(trt_file_path:str, nms_threshold:float=0.35, landmarks:bool=True):
    backend = CenterfaceTRT(trt_file_path)
    engine  = Centerface(backend, landmarks=landmarks)
    engine.prepare(nms=nms_threshold)
    return engine

def get_trt_recognitor(trt_file_path:str):
    engine = RecTRT(trt_file_path)
    engine.prepare()
    return engine

class FaceAnalysis:
    def __init__(self, model_root_path:str, backend:str=Union["trt", "onnx"], 
                 det_onnx:str="", rec_onnx:str="",
                 det_trt:str="", rec_trt:str="", 
                 nmslib_saved_path:str="", l2_threshold:float=1.25,
                 max_size:List[int]=[480, 640], max_rec_batch_size:int=1, 
                 device:str="cuda", landmarks:bool=True, nms_threshold:float=0.6):
        
        
        if backend == "onnx":
            det_onnx_path = os.path.join(model_root_path, det_onnx)
            rec_onnx_path = os.path.join(model_root_path, rec_onnx)
            if not os.path.isfile(det_onnx_path):
                logging.exception(f"det onnx file {det_onnx_path} not found!")
                sys.exit(1)
            if not os.path.isfile(rec_onnx_path):
                logging.exception(f"rec onnx file {rec_onnx_path} not found!")
                sys.exit(1)
            self.det_model = get_onnx_detector(self.det_onnx_path, nms_threshold, landmarks)
            self.rec_model = get_onnx_recognitor(self.rec_onnx_path)
        elif backend == "trt":
            det_trt_path = os.path.join(model_root_path, det_trt)
            rec_trt_path = os.path.join(model_root_path, rec_trt)
            if not os.path.isfile(det_trt_path):
                logging.exception(f"det trt file {det_trt_path} not found!")
                sys.exit(1)
            if not os.path.isfile(rec_trt_path):
                logging.exception(f"rec trt file {rec_trt_path} not found!")
                sys.exit(1)
            self.det_model = get_trt_detector(det_trt_path, nms_threshold, landmarks)
            self.rec_model = get_trt_recognitor(rec_trt_path)
        else:
            logging.info("Unsupported backend type {backend}")
            sys.exit(1)
        self.max_size = max_size # [h, w]
        self.max_rec_batch_size = max_rec_batch_size
        self.device = device
        
        ## create im-memory nms index
        self.l2_threshold = l2_threshold
        
        with open(nmslib_saved_path, "rb") as f:
            embeds, names = pickle.load(f)
        self.names = names
        self.index = nmslib.init(space="l2")
        self.index.addDataPointBatch(embeds)
        index_time_params = {"M":15, "indexThreadQty":4, "efConstruction":100}
        self.index.createIndex(index_time_params)

        
        
    def process_faces(self, faces:List[Face]):
        chunked_faces = to_chunks(faces, self.max_rec_batch_size)
        for chunk in chunked_faces:
            chunk = list(chunk)
            crops = [e.facedata for e in chunk]
            total = len(crops)
            embeddings = [None] * total
            embeddings = self.rec_model.get_embedding(crops)
                
            for i, crop in enumerate(crops):
                embedding = embeddings[i]
                
                normed_embedding = embeddings[i] / norm(embedding)
                face = chunk[i]
                ##we dont need to return face data image
                face = face._replace(facedata=None)
                ##search in-memory index data for the name
                nnInd, nnDist = self.index.knnQuery(normed_embedding, k=1)
                if nnDist > self.l2_threshold:
                    face = face._replace(name="Unknown")
                else:
                    face = face._replace(name=self.names[nnInd[0]])
                #face = face._replace(normed_embedding=normed_embedding)
                yield face

    async def get(self, img:np.ndarray, max_size:List[int]=[480, 640], 
                  prob_threshold:float=0.6):
        
        assert(img.shape[0]==self.max_size[0] and img.shape[1]==self.max_size[1]), f"expect image input size {self.max_size}, but get {img.shape[:2]}"
        
        boxes, probs, landmarks = self.det_model.detect(img, threshold=prob_threshold)
        
        faces = []
        await asyncio.sleep(0)
        if not isinstance(boxes, type(None)):
            for i in range(len(boxes)):
                bbox = boxes[i]
                landmark = landmarks[i]
                _crop = face_align.norm_crop(img, landmark)
                face = Face(bbox=bbox, facedata=_crop)
                faces.append(face)
            faces = [e for e in self.process_faces(faces)]
        return faces
        
