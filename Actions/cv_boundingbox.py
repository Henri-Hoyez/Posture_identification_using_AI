import numpy as np
import cv2

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
        frame = cv2.rectangle(frame, (xMin,yMax) ,(xMax,yMin) , 255,  2)
        result = pose+" : "+str(proba*100)+" %" if isDetected else "Unknown"
        frame = cv2.putText(frame, result ,(xMin+5, yMin+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1,cv2.LINE_AA)

    finally:
        return frame