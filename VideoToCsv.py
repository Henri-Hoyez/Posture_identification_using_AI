import numpy as np
import cv2
from os import walk
import sys
import os
from sys import platform
import argparse
from time import sleep

class VideoToCsv:
    param = dict()

    def __init__(self, inputFolder,outputFolder):
        self.inputFolder = inputFolder
        self.outputFolder = outputFolder
        self.param["model"] = "C:\\Users\\Henri Hoyez\\Desktop\\openpose\\build\\x64\\Release\\models"  # Model directory 

       

    def convertToTrajectory(self,lenght):
        files = []
        for (_, _, filenames) in walk(self.inputFolder):
            files.append(filenames)


        for f in files[0]:
            print(f)
            vidcap = cv2.VideoCapture(self.inputFolder +"\\" + f)
            success,image = vidcap.read()
            count = 0
            success = True
            while success:
                cv2.imwrite("frame%d.jpg" % count, image)    
                success,image = vidcap.read()
                print( 'Read a new frame: ', success)
                count += 1
        