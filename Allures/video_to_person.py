import sys
import cv2
import os
from sys import platform
import argparse
import time
from cv_boundingbox import getBoundingBox
from keras.models import model_from_json
import numpy as np
from queue import Queue
from angles_from_body25 import Geometry_Body25


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
params["number_people_max"] = 1

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


def getNameFromClass(id):
    if id == 0:
        return "Baptiste"
    elif id == 1:
        return "Henri"
    elif id == 2:
        return "Jerome"
    elif id == 3:
        return "Lucas"
    elif id == 4:
        return "Nicolas"
    else:
        return "Unknown"


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# Load model
json_file = open('./models/model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/model3.h5")

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

time_since_lastaction = 0

geometries = Queue(maxsize=15)
currentPerson = None
currentPersonNumber = 0
proba_cum = 0
who = ""
proba_pred = -1
geometry = Geometry_Body25()

if "ip_camera" in params.keys():
    cap = cv2.VideoCapture(params["ip_camera"])
elif "video" in params.keys():
    cap = cv2.VideoCapture(params["video"])
else:
    cap = cv2.VideoCapture(0)


while(cap.isOpened()):
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

        index = 0
        angles = geometry.getAngles(kp[index,:,:])
        distances = geometry.getDistances(kp[index,:,:])


        if geometries.full():
            geometries.get()
        geometries.put(np.append(angles,distances))
        
        if(geometries.full()):
            _input = np.asarray([list(geometries.queue)])
            predictions = model.predict(_input)

            if getNameFromClass(np.argmax(predictions)) != currentPerson:
                currentPerson = getNameFromClass(np.argmax(predictions))
                currentPersonNumber = 1
                proba_cum = predictions[0][np.argmax(predictions)]
                who = ""
                proba_pred = -1
            else:
                currentPersonNumber += 1
                proba_cum += predictions[0][np.argmax(predictions)]

            if currentPersonNumber > 5:
                who = currentPerson
                proba_pred = proba_cum / currentPersonNumber
        frame = getBoundingBox(frame,kp,index,who,proba_pred, True)

        #predictions = model.predict(np.asarray([normalized_keypoints[index,:,:].flatten()]))
        #frame = getBoundingBox(frame,kp,index,getNameFromClass(np.argmax(predictions)),predictions[0][np.argmax(predictions)], isDetected)
    except Exception as e:
        print(e)
        pass

    end = time.time()
    fps = 1 / (end - start)
    # Display the resulting frame
    frame = cv2.putText(frame,'FPS : '+str(int(fps)),(40,40), font, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow('Webcam',frame)

    key = cv2.waitKey(1)
    
    #Wait to press 'q' key for capturing
    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()