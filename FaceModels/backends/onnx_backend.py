import onnxruntime as ort
import cv2
import numpy as np
import logging



class RecONNX:
    def __init__(self, model="/home/justintian/Desktop/FaceRecognition/models/mbface/onnx/mbface.onnx"):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.rec_model = ort.InferenceSession(model, so)
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, ):
        logging.info("warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})
    
    
    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]
        ## use same transform for MBFace and Arcface now
        for i, img in enumerate(face_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            face_img[i] = (img / 127.5 - 1.0)
        face_img = np.stack(face_img)
        net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: face_img})
        return net_out[0]

class CenterfaceONNX:
    def __init__(self, model:str="/home/justintian/Desktop/FaceRecognition/models/centerface/onnx/centerface_rm.onnx"):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.rec_model=ort.InferenceSession(model, so) ##surpassing initializer warnings
        
        self.input = self.rec_model.get_inputs()[0]
        self.output_order = [e.name for e in self.rec_model.get_outputs()]

        self.input_shape = tuple(self.input.shape)

    #warmup
    def prepare(self):
        logging.info("warming up Centerface TensorRT engine...")
        self.rec_model.run(self.output_order,
                {self.rec_model.get_inputs()[0].name:[np.zeros(tuple(self.input.shape[1:]), np.float32)]})
        logging.info(f"Centerface engine warmup complete!")

    def run(self, input):
        net_out = self.rec_model.run(self.output_order, {self.input.name: input})
        return net_out

