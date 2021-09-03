import cv2
import asyncio
import json
import base64
import urllib.request
import numpy as np

import time
import requests


    
## url of IP webcam app running on your smartphone
url = "http://10.0.0.124:8080/shot.jpg"

while True:
    imgResp = urllib.request.urlopen(url)
   
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;) 
    img = cv2.imdecode(imgNp,-1)
    string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    payload = json.dumps({"img":string, "username":"me"})
    response = requests.put("http://127.0.0.1:8080/predict", data=payload)
    res_dict = response.json()
    
    for face in res_dict["faces"]:
        name = face["name"]      
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(img, name, (int(x1 + 6), int(y2 - 6)), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
 
    # put the image on screen
    cv2.imshow('IPWebcam',img)

    #To give the processor some less stress
    time.sleep(0.02) 

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
