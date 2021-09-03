import traceback
from typing import List

import numpy as np

from app.db.db_client import AsyncIOMotorClient
from app.db.db_config import database_name, face_collection_name


async def create_face(conn:AsyncIOMotorClient, identity:str, embed:List[float], imgID:str):
    tb = None
    try:
        row = await conn[database_name][face_collection_name].insert_one({"imgID" : imgID,
                                                                          "identity":identity,
                                                                          "embed":embed})
    except Exception as e:
        tb = traceback.format_exc()
    return tb

async def find_nn(conn:AsyncIOMotorClient, target_embed:List[float], threshold:float=1.25):
    pipeline = [
                    {
                        "$addFields":{"target_embed":target_embed}
                    },
                    {"$unwind" : {"path" : "$embed", "includeArrayIndex": "embed_index"}},
                    {"$unwind" : {"path" : "$target_embed", "includeArrayIndex": "target_index"}},
                    {
                        "$project":{
                            "imgID" : 1,
                            "identity":1,
                            "embed": 1,
                            "target_embed":1,
                            "compare": {
                                "$cmp":["$embed_index", "$target_index"]
                            }
                        }
                    },
                    {
                        "$group":{
                            "_id": "$imgID",
                            "distance":{
                                "$sum": {
                                    "$pow": [{
                                        "$subtract": ["$embed", "$target_embed"]
                                    }, 2]
                                }
                            }
                        }
                    },
                    {
                        "$project":{
                            "_id":1,
                            "identity":1,
                            "distance":1,
                            "cond":{"$lte": [ "$distance", threshold]}
                        }
                    },
                    {"$match": {"cond":True}},
                    {"$sort" : {"distance" : 1}}
            ]
    query = conn[database_name][face_collection_name].aggregate(pipeline)
    return query
