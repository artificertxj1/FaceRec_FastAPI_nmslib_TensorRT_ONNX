from typing import Dict, List, Optional, Union
import time
import traceback
import os
import sys
import logging
import base64
import io
import json

import numpy as np
import cv2

from FaceModels.face_model import FaceAnalysis, Face
from FaceModels.utils.pad_image import pad_image





async def decode_image(data:str=None):
    img_data = dict(data=None,
                    traceback=None)
    b64encoded = data
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        __bin = np.fromstring(__bin, np.uint8)
        image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
    except Exception:
        tb = traceback.format_exc()
        img_data.update(traceback=tb)
        return img_data
    img_data.update(data=image)
    return img_data



def serialize_face(face:Face):
    _face_dict = dict(bbox=None, name=None)

    if face.name is not None:
        _face_dict.update(name=face.name)
    if face.bbox is not None:
        _face_dict.update(bbox=face.bbox.astype(int).tolist())
   
    return _face_dict

class Processing:
    def __init__(self, config:dict):
        
        model_root_path = config["model_root_path"]
        backend  = config["backend"]
        det_onnx = config["det_onnx"]
        rec_onnx = config["rec_onnx"]
        det_trt  = config["det_trt"]
        rec_trt  = config["rec_trt"]
        self.max_size = config["max_size"]
        self.max_rec_batch_size = config["max_rec_batch_size"]
        device = config["device"]
        landmarks = config["landmarks"]
        nms_threshold = config["nms_threshold"]
        nmslib_saved_path = config["nmslib_saved_path"]
        l2_threshold = config["l2_threshold"]
        self.model = FaceAnalysis(model_root_path, backend, det_onnx, 
                rec_onnx, det_trt, rec_trt, 
                nmslib_saved_path, l2_threshold,
                self.max_size, self.max_rec_batch_size, device, 
                landmarks, nms_threshold)

    async def embed(self, img:str):
        output = dict(status=None, traceback=None, faces=[])
        img_data = await decode_image(img)
        try:
            if img_data["traceback"] is not None:
                output["status"] = "error"
                output["traceback"] = img_data.get("traceback")
            else:
                img = img_data.get("data")
                if img.shape[0] != self.max_size[0] or img.shape[1] != self.max_size[1]:
                    img = cv2.resize(img, (self.max_size[1], self.max_size[0]),
                                         interpolation=cv2.INTER_AREA)	
                faces = await self.model.get(img)
                for face in faces:
                    face_dict = serialize_face(face)
                    output["faces"].append(face_dict)
                output["status"] = "ok"
        except Exception as e:
            tb = traceback.format_exc()
            output["status"] = "error"
            output["traceback"] = tb
        return output
		
    async def embed_arb_shape(self, img:str):
        #only takes one picture
        #if multiple faces presented, only return embedding vector of one face
        output = dict(status="error", traceback=None, embed=None)
        img_data = await decode_image(img)
        try:
            if img_data["traceback"] is not None:
                output["status"] = "error"
                output["traceback"] = img_data.get("traceback")
            else:
                img = img_data.get("data")
                img = pad_image(img, max_size=self.max_size)
                faces = await self.model.get(img)
                for face in faces:
                    if face.normed_embedding is not None and output["embed"] is None:
                        output["embed"] = face.normed_embedding.tolist()
                        break
            output["status"] = "ok"
        except Exception as e:
            tb = traceback.format_exc()
            output["status"] = "error"
            output["traceback"] = tb
        return output
    
