# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
from tqdm import tqdm
import time
import datetime
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
parser.add_argument("--source", default="./datasets_videos", help="Choose the directory source of the videos")
parser.add_argument("--dest", default="./datasets_generated", help="Choose the directory destination of the CSV")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["number_people_max"] = 1

source = args[0].source
destination = args[0].dest


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



# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

data = []
DATASET_NAME = "dataset_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

my_paths = dict()
isFirst = True
generated_path=destination
for root, dirs, files in os.walk(source):
    if isFirst:
        isFirst=False
        continue
    my_paths[root]=[]
    for file in files:
        try:
            os.makedirs(os.path.join(generated_path,os.path.basename(root)))
        except:
            pass
        finally:
            my_paths[root].append(file)

id_class = dict()
id_cum = 0
geometry = Geometry_Body25()

for path in my_paths.keys():
    class_data = os.path.basename(path)
    if not class_data in id_class.keys():
        id_class[class_data] = id_cum
        id_cum += 1

    for file in my_paths[path]:
        dataset_name = os.path.join(os.path.join(generated_path,os.path.basename(path)),"dataset_"+file.replace(".","_")+".csv")
        DATASET_FILE = open(dataset_name,"w+")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        f = os.path.join(path,file)
        f = os.path.relpath(f, os.path.commonprefix([f, dir_path]))
        vidcap = cv2.VideoCapture(f)
        success,image = vidcap.read()
        datum = op.Datum()
        TOTAL_FRAME_COUNT = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video : "+f)
        pbar = tqdm(total=TOTAL_FRAME_COUNT)
        success = True
        while success:
            try:
                success,image = vidcap.read()
                datum.cvInputData = image
                opWrapper.emplaceAndPop([datum])
                
                pbar.update(1)
                kp = datum.poseKeypoints

                if len(kp.shape) == 0:
                    continue
                
                class_data = os.path.basename(path)


                tmp = ""
                tmp += f.replace(";","") + ";"
                tmp += class_data + ";"
                tmp += str(id_class[class_data]) + ";"
                tmp += ",".join([str(elem) for elem in geometry.getAngles(kp[0,:,:])]) + ";"
                tmp += ",".join([str(elem) for elem in geometry.getDistances(kp[0,:,:])])
                DATASET_FILE.write(tmp+"\n")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit()
            except Exception as e:
                print("******************************************")
                print("* error in "+f)
                print(e)
                print("******************************************")

                break
        pbar.close()