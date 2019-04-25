import numpy as np
import cv2
import traceback
from random import randint


def getBoundingBox(frame, keypoints, index, pose, proba, isDetected):
    try:
        table = keypoints[index,:,:]
        xMin=table[:,0][table[:,0] != 0].min()
        xMax=table[:,0][table[:,0] != 0].max()
        yMax=table[:,1][table[:,1] != 0].max()
        yMin=table[:,1][table[:,1] != 0].min()
        xMin=int(xMin-((10/100)*(xMax-xMin)))
        xMax=int(xMax+((10/100)*(xMax-xMin)))
        yMin=int(yMin-((10/100)*(yMax-yMin)))
        yMax=int(yMax+((10/100)*(yMax-yMin)))

        result = pose if isDetected else "Unknown"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .5
        rectangle_bgr = (180, 255, 50)
        text_color = (255, 255, 255)
        (text_width, text_height) = cv2.getTextSize(result, font, fontScale=font_scale, thickness=1)[0]

        text_offset_x = xMin+4
        text_offset_y = yMin-5

        box_coords = ((text_offset_x-5, text_offset_y+5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))

        frame = cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        frame = cv2.putText(frame, result, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)

        xMax = xMax if xMax - xMin > text_width + 10 else xMin+text_width+9
        
        frame = cv2.rectangle(frame, (xMin,yMax) ,(xMax,yMin) , (180,255,50),  2)

    except Exception as e:
        try:
            exc_info = sys.exc_info()
        finally:
            traceback.print_exception(*exc_info)
            del exc_info
    finally:
        return frame