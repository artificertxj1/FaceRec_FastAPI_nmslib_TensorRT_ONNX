import os
import cv2
import numpy as np
import time
import logging


from FaceModels.backends.trt_loader import TrtModel


class RecTRT:
    def __init__(self, rec_name:str="/home/justintian/Desktop/FaceRecognition/models/mbface/trt/mbface.trt"):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None
        self.max_batch_size = 16
        
    def prepare(self):
        logging.info("warming up Arcface TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]
            
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Arcface engine warmup complete!")

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]
        if not face_img[0].shape == (3, 112, 112):
            for i, img in enumerate(face_img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = img.astype(np.float32)
                face_img[i] = (img / 127.5 - 1.0)
            face_img = np.stack(face_img)
        embeddings = self.rec_model.run(face_img, deflatten=True)[0]
        return embeddings
    
class CenterfaceTRT:
    def __init__(self, model:str="/home/justintian/Desktop/FaceRecognition/models/centerface/trt/centerface.trt"):
        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)
        self.input_shape = None
        self.output_order = None
        
    #warmup
    def prepare(self):
        logging.info("warming up Centerface TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.output_order = self.rec_model.out_names
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Centerface engine warmup complete!")
        
    def run(self, input):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True)
        net_out = [net_out[e] for e in self.output_order]
        return net_out
