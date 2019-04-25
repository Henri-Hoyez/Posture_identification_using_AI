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
parser.add_argument("--df", default="./datasets", help="Asks for the path folder where the user wants to store the dataset (Only available for pose recognition)")
parser.add_argument("--mode", default="webcam", help="Mode of recognition")
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


# Looks if the program in in developer mode
args = args[0].__dict__
if "dev_mode" in args.keys() and args["mode"] == "pose":
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

# Print or not some help
display_help = True


#Choose the camera to work on
if "ip_camera" in params.keys():
    cap = cv2.VideoCapture(params["ip_camera"])
elif "video" in params.keys():
    cap = cv2.VideoCapture(params["video"])
else:
    cap = cv2.VideoCapture(0)
    print("Let's go !")

#Set the title of the window
if args["mode"] == "pose":
    title_window = "Pose Recognition"
elif args["mode"] == "action":
    title_window = "Action Recognition"
elif args["mode"] == "gait":
    title_window = "Gait Recognition"
else:
    title_window = "Webcam"

cv2.namedWindow(title_window, cv2.WINDOW_AUTOSIZE)

while(cap.isOpened()):

    start = time.time()
    ret, frame = cap.read()

    if title_window != "Webcam":

        if isInDevMode and isSavingData and len(dataset) == 100:
            dataset_manager.save(dataset)
            dataset = []

        try:
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

        #Normalize keypoints and predict pose
        try:
            normalized_keypoints = np.array(kp)
            for index in range(normalized_keypoints.shape[0]):
                
                #Normalize keypoints
                min_X = min(normalized_keypoints[index,:,0])
                max_X = max(normalized_keypoints[index,:,0])
                min_Y = min(normalized_keypoints[index,:,1])
                max_Y = max(normalized_keypoints[index,:,1])
                min_Z = min(normalized_keypoints[index,:,2])
                max_Z = max(normalized_keypoints[index,:,2])
                normalized_keypoints[index,:,0] = (normalized_keypoints[index,:,0]-min_X)/(max_X-min_X)
                normalized_keypoints[index,:,1] = (normalized_keypoints[index,:,1]-min_Y)/(max_Y-min_Y)
                normalized_keypoints[index,:,2] = (normalized_keypoints[index,:,2]-min_Z)/(max_Z-min_Z)

                # Check that the necessary points are detected
                isDetected = (kp[index,11,:].sum() != 0)
                isDetected = (isDetected and (kp[index,14,:].sum() != 0))
                isDetected = (isDetected and (kp[index,8,:].sum() != 0))
                isDetected = (isDetected and (kp[index,1,:].sum() != 0))

                # Keep in memory the history of images
                if isInDevMode and isSavingData and isDetected and normalized_keypoints.shape[0] < 4:
                    new_data = ""
                    new_data += "None;"
                    new_data += str(mode.name) + ";"
                    new_data += str(mode.value) + ";"
                    new_data += ",".join(map(str,list(np.asarray(normalized_keypoints[index,:,:].flatten()))))
                    dataset.append(new_data)

                # Predict the pose
                predictions = model.predict(np.asarray([normalized_keypoints[index,:,:].flatten()]))

                # Dispaly the prediction
                frame = getBoundingBox(frame,kp,index,getNameFromClass(np.argmax(predictions)),predictions[0][np.argmax(predictions)], isDetected)
        except:
            pass

        # Calculate the frame rate
        end = time.time()
        fps = 1 / (end - start)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    # Get window size
    x,y,w,h = cv2.getWindowImageRect(title_window)

    # Display the title on the window
    title = title_window
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    (text_width, text_height) = cv2.getTextSize(title, font, fontScale=font_scale, thickness=1)[0]
    img = frame.copy()
    img = cv2.rectangle(img, (int(w/2-text_width/2-10),text_height+45), (int(w/2+text_width/2+5),20), (255,255,255), cv2.FILLED)
    frame = cv2.addWeighted(img, .3, frame, .7, 0)
    frame = cv2.putText(frame,title,(int(w/2-text_width/2),text_height+30), font, font_scale,(20,30,0),2,cv2.LINE_AA)

    # Display the frame rate on the window
    text_fps = 'FPS : '+str(int(fps))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .8
    (text_width, text_height) = cv2.getTextSize(text_fps, font, fontScale=font_scale, thickness=1)[0]
    img = frame.copy()
    img = cv2.rectangle(img, (w-text_width-30,h-10), (w-10,h-text_height-30), (255,255,255), cv2.FILLED)
    frame = cv2.addWeighted(img, .5, frame, .5, 0)
    frame = cv2.putText(frame,text_fps,(w-text_width-20,h-20), font, font_scale,(20,30,0),2,cv2.LINE_AA)

    # Display the help on the window for dev mode
    if display_help:
        text = "'h' : Display/Hide Help"
        if isInDevMode:
            text = text + "\n'd' : Begin saving standing data\n'a' : Begin saving sitting data\n'c' : Begin saving laying data\n's' : Stop saving data"
        text = text + "\n'q' : Quit application"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .3
        nb_lines = len(text.split("\n"))
        max_width = 0
        text_height = 0
        sum_heights = 0
        lines = text.split("\n")
        datas = []
        for i in range(nb_lines):
            (text_width, text_height) = cv2.getTextSize(lines[i], font, fontScale=font_scale, thickness=1)[0]
            sum_heights += text_height + 10
            max_width = text_width if text_width > max_width else max_width
            datas.append([text_width,text_height])
        img = frame.copy()
        img = cv2.rectangle(img, (5,int(h/2+sum_heights/2+10)), (max_width+30,int(h/2-sum_heights/2-10)), (255,255,255), cv2.FILLED)
        frame = cv2.addWeighted(img, .5, frame, .5, 0)
        for i in range(len(lines)):
            frame = cv2.putText(frame,lines[i],(15,int(h/2-sum_heights/2+(i+1)*(text_height+5))), font, font_scale,(20,30,0),1,cv2.LINE_AA)

    # Set visible the new computed image
    cv2.imshow(title_window,frame)

    #Wait to press 'h' key for displaying help for dev mode
    key = cv2.waitKey(33)
    if key == ord('h'):
        print("Press h")
        display_help = not display_help

    #Wait to press 'd' key for capturing data "standing"
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