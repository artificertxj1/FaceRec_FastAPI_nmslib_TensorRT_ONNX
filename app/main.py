import sys
sys.path.append("../")
import os
import pickle
import json
import traceback
import logging 
import time
from typing import Optional, List

import nmslib
from fastapi import FastAPI, File, UploadFile
from fastapi import Depends
from pydantic import BaseModel, Field, validator

from FaceModels.processing import Processing


## create global models
config_path = "/home/justintian/Desktop/FaceRecognition/app/config.json"
try:
    with open(config_path, "r") as f:
            config = json.load(f)
except Exception as e:
    print(f"Failed to load config file, exception {e}")
    sys.exit(1)
model = Processing(config=config["model_configs"])





app = FastAPI(
    title="FaceDetAndRec-REST",
    description="RESTful API of trt face models",
    version="0.1.0"
)


class Input(BaseModel):
    img:str
    name:Optional[str] = None
    username:Optional[str] = None
    imgID:Optional[str] = None
    
    @validator("name")
    def prevent_none_unknown(cls, v):
        assert v is not None, "name may not be None"
        assert v.lower()!="unknown", "name may not be 'unknown'"
        return v
    @validator("imgID")
    def prevent_none(cls, v):
        assert v is not None, "imgpath may not be None"
        return v

@app.put("/predict", tags=["Detection & Recognition"])
async def predict(data:Input):
    img, username = data.img, data.username
    response = await model.embed(img)
    return response


    
    
    
