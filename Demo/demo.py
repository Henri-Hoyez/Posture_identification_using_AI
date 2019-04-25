import sys
import cv2
import os
from sys import platform
import argparse
import time
import traceback
import numpy as np
from classes import Pose, Action, Gait
from PredictionService import PredictionService
from DatasetManager import DatasetManager
from main_layout import setMainLayout
import parse_args
from skeleton import Skeleton
import datetime

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
args, params = parse_args.GetParameters()

# Looks if the program in in developer mode
if "dev_mode" in args.keys() and args["mode"] == "pose":
    isInDevMode = args["dev_mode"]
else:
    isInDevMode = False
if isInDevMode and "df" in args.keys():
    try:
        os.makedirs(args["df"])
    except:
        pass
    dataset_manager = DatasetManager(args["df"])
else:
    dataset_manager = None
isSavingData = False
mode = Pose.unknown
dataset = []


# Print or not some help
display_help = True

#View skeleton or not
if "no-skeleton" in args.keys() and args["no-skeleton"]:
    viewSkeleton = False
else:
    viewSkeleton = True

#Choose the camera to work on
thermical = False
if "ip_camera" in params.keys():
    if "thermic" in args.keys() and args["thermic"]:
        thermical = True
    cap = cv2.VideoCapture(params["ip_camera"])
elif "video" in params.keys():
    cap = cv2.VideoCapture(params["video"])
else:
    cap = cv2.VideoCapture(0)


#Set the title of the window and load associated model
if args["mode"] == "pose":
    title_window = "Pose Recognition"
    model = "model_pose"
    if "thermic" in args.keys() and args["thermic"]:
        params["number_people_max"] = 1
elif args["mode"] == "action":
    title_window = "Action Recognition"
    model = "model_action"
    params["number_people_max"] = 1
elif args["mode"] == "gait":
    title_window = "Gait Recognition"
    model = "model_gait"
    params["number_people_max"] = 1
elif args["mode"] == "skeleton":
    title_window = "Skeleton view"
    model = None
    #params["number_people_max"] = 1
elif args["mode"] == "openpose":
    title_window = "Openpose"
    model = None
else:
    title_window = "Webcam"
    model = None


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

service = PredictionService(args["mode"], model)
skeleton = Skeleton(color_joints=(0,150,50))

cv2.namedWindow(title_window, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(title_window, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# Get window size
x,y,w,h = cv2.getWindowImageRect(title_window)


try:
    os.makedirs(args["out"])
except:
    pass
if "out" in args.keys() and args["out"] != None:
    out = cv2.VideoWriter(os.path.join(args["out"],"Recording_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"),cv2.VideoWriter_fourcc(*'MP4V'), 13 if thermical else 17, (w,h))
else:
    out = None
print(out)
while(cap.isOpened()):

    start = time.time()
    ret, frame = cap.read()

    

    if title_window != "Webcam":

        if isInDevMode and isSavingData and len(dataset) >= 100:
            dataset_manager.save(dataset)
            dataset = []

        if thermical:
            frame = cv2.resize(frame,None,fx=6,fy=6)
        try:
            # Process Image
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
        except:
            break

        #Get keypoints
        kp = datum.poseKeypoints
        
        if args["mode"] == "skeleton":
            frame = np.zeros(frame.shape,np.uint8)
            frame += 200
            try:
                frame = skeleton.drawSkeleton(kp,(x,y,w,h),frame)
            except:
                pass

        else:
            #Add skeleton
            if viewSkeleton:
                frame = datum.cvOutputData

            #Normalize keypoints and predict pose
            if args["mode"] != "openpose":
                try:
                    frame = service.displayPrediction(frame, kp)
                    if args["mode"] == "pose" and isInDevMode and isSavingData:
                        dev_data = service.getData()
                        for data in dev_data:
                            dataset.append("None;" + str(mode.name) + ";" + str(mode.value) + ";" + data)
                except Exception as e:
                    try:
                        exc_info = sys.exc_info()
                    finally:
                        traceback.print_exception(*exc_info)
                        del exc_info

        # Calculate the frame rate
        fps = 1 / (time.time() - start)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)


    # Display the title and the frame rate on the window
    # Display the help on the window for dev mode
    frame = setMainLayout((x,y,w,h),frame,title_window,fps,display_help,isInDevMode)
    
    if out != None:
        out.write(frame)

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
if out != None:
    out.release()
cap.release()
cv2.destroyAllWindows()