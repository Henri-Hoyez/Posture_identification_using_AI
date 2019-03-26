# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import pandas
from tqdm import tqdm
import datetime


def getclassFromName(fileName):

    if (fileName.find("laying")):
        return 1
    
    if(fileName.find("sitting")):
        return 2
    
    if(fileName.find("walking")):
        return 3

 



# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["video"] = "C:\\Users\\Henri Hoyez\\Desktop\\walking.mp4"


# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

data = []
FRAME_LENGHT = 10
day =str(datetime.datetime.today().day)
month = str(datetime.datetime.today().month)
year = str(datetime.datetime.today().month)
year = str(datetime.datetime.today().month)
INPUT_PATH = "C:\\Users\\Henri Hoyez\\Desktop\\Posture_identification_using_AI\\LSTM\\Dataset\\Video"          # Video inputs
OUTPUT_PATH = "C:\\Users\\Henri Hoyez\\Desktop\\Posture_identification_using_AI\\LSTM\\Dataset\\TrainData"     # Dataset output path
DATASET_NAME = "dataset.csv"

DATASET_FILE = open(DATASET_NAME,"a+")


test={}
test["class_id"]=[]
test["pts"]=[]
TOTAL_FRAME_COUNT = 0
count = 0





files = []
for (_, _, filenames) in os.walk(INPUT_PATH):
    files.append(filenames)
    


for f in files[0]:
    cap = cv2.VideoCapture(INPUT_PATH + "\\" + f)
    TOTAL_FRAME_COUNT += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pbar = tqdm(total=TOTAL_FRAME_COUNT)

for f in files[0]:
    vidcap = cv2.VideoCapture(INPUT_PATH + "\\" + f)
    success,image = vidcap.read()
    datum = op.Datum()

    success = True
    while success:
        try:
            success,image = vidcap.read()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])

            cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
            count += 1
            pbar.update(1)
            kp = datum.poseKeypoints

            normalized_keypoints = np.array(kp)

            for index in range(normalized_keypoints.shape[0]):
                #display
                min_X = min(normalized_keypoints[index,:,0])
                max_X = max(normalized_keypoints[index,:,0])
                min_Y = min(normalized_keypoints[index,:,1])
                max_Y = max(normalized_keypoints[index,:,1])
                min_Z = min(normalized_keypoints[index,:,2])
                max_Z = max(normalized_keypoints[index,:,2])
                normalized_keypoints[index,:,0] = (normalized_keypoints[index,:,0]-min_X)/(max_X-min_X)
                normalized_keypoints[index,:,1] = (normalized_keypoints[index,:,1]-min_Y)/(max_Y-min_Y)
                normalized_keypoints[index,:,2] = (normalized_keypoints[index,:,2]-min_Z)/(max_Z-min_Z)

                isDetected = (kp[index,11,:].sum() != 0)
                isDetected = (isDetected and (kp[index,14,:].sum() != 0))
                isDetected = (isDetected and (kp[index,8,:].sum() != 0))
                isDetected = (isDetected and (kp[index,1,:].sum() != 0))


            tmp = str(getclassFromName(f))+";"
            tmp+= ",".join(map(str,list(np.asarray(normalized_keypoints[index,:,:].flatten()))))
            DATASET_FILE.write(tmp+"\n")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        except TypeError as e:
            break

    #new_data += ",".join(map(str,list(np.asarray(normalized_keypoints[index,:,:].flatten()))))

    
    

    
