import sys
sys.path.append("../")
import os
import json
import traceback
import logging 
import time
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile
from fastapi import Depends
from pydantic import BaseModel, Field, validator

from pymongo import MongoClient

from FaceModels.processing import Processing

from app.db.db_client import AsyncIOMotorClient
from app.db.db_client import get_database, connect_to_mongo, close_mongo_connection
from app.db.db_crud import create_face

## create global models
config_path = "/home/justintian/Desktop/FaceRecognition/app/config.json"
try:
    with open(config_path, "r") as f:
            config = json.load(f)
except Exception as e:
    print(f"Failed to load config file, exception {e}")
    sys.exit(1)


model = Processing(config=config["model_configs"])

#print("connecting...")
#client = MongoClient(config["mongo_configs"]["db_host"], config["mongo_configs"]["db_port"])
#face_db = client[config["mongo_configs"]["db_name"]]
#print("connected!")

#def insert_face(name:str, embed:List[float]):
#    res = face_db[config["mongo_configs"]["collection_name"]].insert_one({"name":name, "embed":embed})
#    print("result %s"%repr(res.inserted_id))

app = FastAPI(
    title="FaceDetAndRec-REST",
    description="RESTful API of trt face models",
    version="0.1.0"
)

app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

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
    img, username = data.img
    response = await model.embed(img)
    return response

@app.put("/add", tags=["Add a new identity to the database"])
async def add(data:Input, db:AsyncIOMotorClient=Depends(get_database)):
    response = {"status":"ok", "traceback":None}
    embed, tb = None, None
    try:
        face_name = data.name
        face_img  = data.img
        imgID     = data.imgID
        embed_res = await model.embed_arb_shape(face_img)
        if embed_res["traceback"] is None:
            embed = embed_res["embed"]
        else:
            response["status"] = "error"
            response["traceback"] = embed_res["traceback"]
            return response
        if embed is not None:
            tb = await create_face(db, face_name, embed, imgID)
        if tb is not None:
            response["traceback"]= tb
            response["status"] = "error"
    except Exception as e:
        tb = traceback.format_exc()
        response["status"] = "error"
        response["traceback"] = tb
    return response
    
    
    
