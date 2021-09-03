from typing import List
import numpy as np
import cv2

def pad_image(img:np.ndarray, max_size:List[int]=[480, 640]):
    h, w, _ = img.shape
    cw, ch = max_size[1], max_size[0]
    scale_factor = min(float(cw) / w, float(ch/h))
    transformed_image = cv2.resize(img, (0, 0), fx=scale_factor, 
                                 fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    h, w, _ = transformed_image.shape
    if w < cw:
        transformed_image = cv2.copyMakeBorder(transformed_image, 
                                               0, 0, 0, cw-w,
                                               cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image,
                                               0, ch-h, 0, 0,
                                               cv2.BORDER_CONSTANT)
    return transformed_image
