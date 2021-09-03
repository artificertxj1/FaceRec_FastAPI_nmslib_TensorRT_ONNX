# FaceRecognition

I tried to make a face recognition system in this repo.

Current inferencing backend supports ONNX and TensorRT models. The first attempt is to saving face embedding vectors into a MongoDB collection and query the nearest neighbor of a detected face from the collection. I tried to write an aggregate query (check db_crud.py) but couldn't get correct results from it. I cannot find an off-shelf solution of high dimension vector query in database. Postgresql CUBE method only supports an up to 100 dimension vector. There are ways to work around this limit and use a 128-d vector. But it still won't work with any over 200-d vectors. Most recent face recognition models like arcface or cosface embeds a face picture into a 512-d vec which is out the capability of Postgresql.

Optimizing a database engine to fit this certain task is beyond my reach now. So, I will just leave my result codes here. I put a sample result using a pickle file as a small in-memory database. The distance between the detected face and saved face is calculated by numpy. If you like to do this by yourself, just change face_model.py by adding saved_embeds as a class variable.  

Sorry for the mess of the codes.  If someone knows how to do a 512-d vector nearest neighbor search, pls let me know by making an issue here.
