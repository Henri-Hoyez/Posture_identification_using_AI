import sys
import cv2
import os
from sys import platform
import argparse
import time
from cv_boundingbox import getBoundingBox
from keras.models import model_from_json
import numpy as np
from pose import Pose
from DatasetManager import DatasetManager

def getClassFromName(fileName):
    if (fileName.find("laying")):
        return 0
    if(fileName.find("sitting")):
        return 1
    if(fileName.find("standing")):
        return 2
    return None

def getNameFromClass(id):
    if id == 0:
        return "laying"
    elif id == 1:
        return "sitting"
    elif id == 2:
        return "standing"
    else:
        return "Unknown"

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../../x64/Release;' +  dir_path + '/../../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--dev_mode", action='store_true', help="Enter in developer mode")
parser.add_argument("--df", default="./datasets", help="Asks for the path folder where the user wants to store the dataset")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../../models/"

# Add others in path
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

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# Load model
json_file = open('./models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/model.h5")

font = cv2.FONT_HERSHEY_SIMPLEX

args = args[0].__dict__

# dataset generation variables
if "dev_mode" in args.keys():
    isInDevMode = args["dev_mode"]
else:
    isInDevMode = False

if isInDevMode and "df" in args.keys():
    dataset_manager = DatasetManager(args["df"])
else:
    dataset_manager = None

isSavingData = False
mode = Pose.unknown
dataset = []



if "ip_camera" in params.keys():
    cap = cv2.VideoCapture(params["ip_camera"])
else:
    cap = cv2.VideoCapture(0)


while(True):
    if isInDevMode and isSavingData and len(dataset) == 100:
        dataset_manager.save(dataset)
        dataset = []

    start = time.time()

    # Capture frame-by-frame
    try:
        ret, frame = cap.read()
        # Process Image
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
    except:
        break

    #Get keypoints
    kp = datum.poseKeypoints
    
    #Add skeleton
    frame = datum.cvOutputData

    try:
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

            if isInDevMode and isSavingData and isDetected and normalized_keypoints.shape[0] < 4:
                new_data = ""
                new_data += "None;"
                new_data += str(mode.name) + ";"
                new_data += str(mode.value) + ";"
                new_data += ",".join(map(str,list(np.asarray(normalized_keypoints[index,:,:].flatten()))))
                dataset.append(new_data)

            predictions = model.predict(np.asarray([normalized_keypoints[index,:,:].flatten()]))
            frame = getBoundingBox(frame,kp,index,getNameFromClass(np.argmax(predictions)),predictions[0][np.argmax(predictions)], isDetected)
    except:
        pass

    end = time.time()
    fps = 1 / (end - start)
    # Display the resulting frame
    frame = cv2.putText(frame,'FPS : '+str(int(fps)),(40,40), font, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow('Webcam',frame)

    #Wait to press 'd' key for capturing data "standing"
    key = cv2.waitKey(33)
    if key == ord('d') and isInDevMode:
        print("Press d")
        if not isSavingData:
            isSavingData = True
            mode = Pose.standing

    #Wait to press 'a' key for capturing data "sitting"
    if key == ord('a') and isInDevMode:
        print("Press a")
        if not isSavingData:
            isSavingData = True
            mode = Pose.sitting

    #Wait to press 'c' key for capturing data "laying"
    if key == ord('c') and isInDevMode:
        print("Press c")
        if not isSavingData:
            isSavingData = True
            mode = Pose.laying

    #Wait to press 's' key for stopping capturing data
    if key == ord('s') and isInDevMode:
        print("Press s")
        isSavingData = False
        mode = Pose.unknown
        dataset_manager.save(dataset)
        dataset = []

    #Wait to press 'q' key for capturing
    if key == ord('q'):
        if isSavingData:
            dataset_manager.save(dataset)
            dataset = []
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

