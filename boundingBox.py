import numpy as np
import cv2

def getBoundingBox(frame, table):
    xMin=table[0,:].min()
    xMax=table[0,:].max()
    yMax=table[1,:].max()
    yMin=table[1,:].min()
    xMin=int(xMin-((10/100)*(xMax-xMin)))
    xMax=int(xMax+((10/100)*(xMax-xMin)))
    yMin=int(yMin-((10/100)*(yMax-yMin)))
    yMax=int(yMax+((10/100)*(yMax-yMin)))
    return cv2.rectangle(frame, (xMin,yMax) ,(xMax,yMin) , 255,  2)