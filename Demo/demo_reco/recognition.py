from skeleton import Skeleton
import cv2
import sys
import os
from sys import platform
import argparse
import time
import datetime
from keras.models import model_from_json
import numpy as np
from GaitClassifier import GaitClassifier
from classes import Gait


class recognitionGui:
    def __init__(self,video_name):
        self.title_window="Demonstrateur reconnaissance de personne"
        self.skeletonService=Skeleton()
        self.cap = cv2.VideoCapture(video_name)
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.gaitClassifier=GaitClassifier()
        self.last_frame=None
        self.kp_frames=[]
        self.stop =False
        w, h = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.out = cv2.VideoWriter("Recording_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4",cv2.VideoWriter_fourcc(*'MP4V'), 25, (w,h))

    def run(self):
        while(self.cap.isOpened() and not self.stop):
            self.putFrame()
        
        cv2.imshow(self.title_window,self.last_frame)
        self.out.write(self.last_frame)

        self.kp_frames=np.asarray(self.kp_frames)
        print(self.kp_frames.shape)
        if self.kp_frames.shape[0]>=140:
            for i in range(3):
                mean = np.mean(self.kp_frames[:,i::3],axis = 0)
                std = np.std(self.kp_frames[:,i::3],axis = 0)
                self.kp_frames[:,i::3] = np.divide(self.kp_frames[:,i::3]-mean,std,where=(std!=0))
            pred, pred_value=self.gaitClassifier.demo(self.kp_frames)
            pred=Gait(pred).name
        else:
            pred="Unknown"
            pred_value = None
        self.last_frame = self.putPhoto("people/"+pred+".png",self.last_frame)
        self.last_frame=self.putName(pred,self.last_frame)
        if pred_value != None:
            self.last_frame=self.putName(str(int(pred_value * 10000) / 100) + " %",self.last_frame,offset = 15)
        cv2.imshow(self.title_window,self.last_frame)

        for i in range(25):
            self.out.write(self.last_frame)
        self.out.release()
        cv2.waitKey(0)


    def putFrame(self):
        
        ret, frame = self.cap.read()
        if ret:
            try:
                # Process Image
                datum = op.Datum()
                datum.cvInputData = frame
                self.opWrapper.emplaceAndPop([datum])
            except:
                pass

            kp = datum.poseKeypoints
            if not len(kp.shape)==0:
                self.kp_frames.append(kp[0].flatten())
            
            frame = self.putSkeleton(kp, frame)
            frame = self.putPhoto("people/Unknown.png",frame)
            self.last_frame=np.array(frame)
            frame=self.putName(None,frame)
            cv2.imshow(self.title_window,frame)
            self.out.write(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.stop = True
        else:
            self.cap.release()


    def putSkeleton(self,kp, frame):
        img = np.zeros(frame.shape,np.uint8)
        img += 180

        if not len(kp.shape) == 0 :
            window_size=[frame.shape[0],frame.shape[1],50,90]
            img=self.skeletonService.drawSkeleton(kp,window_size,img)
            
        img=cv2.resize(img,(int(int(1/3*frame.shape[0]) * 4/3),int(1/3*frame.shape[0])))
        img_w, img_h, _ = img.shape
        frame[frame.shape[0] - img_w - 10:frame.shape[0] - 10,frame.shape[1] - img_h - 10:frame.shape[1] - 10] = img
        return frame

    def putPhoto(self, path, frame):
        photo = cv2.imread(path)
        photo = cv2.resize(photo,(int(1/3*frame.shape[0]),int(1/3*frame.shape[0])))            

        photo_w, photo_h, _ = photo.shape
        frame[10:photo_w + 10,10:photo_h + 10] = photo
        return frame

    def putName(self, name, frame, offset = 0):
        
        if name == None:
            name = "Reconnaissance en cours ..."
            font_scale = .3
        else:
            font_scale = .6 if offset == 0 else 0.45
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height) = cv2.getTextSize(name, font, fontScale=font_scale, thickness=1)[0]
    
        img = frame.copy()
        img = cv2.rectangle(img, (10,int(offset+11+text_height+1/3*frame.shape[0])), (int(10+1/3*frame.shape[0]),int(offset+10+1/3*frame.shape[0])), (255,255,255), cv2.FILLED)
        frame = cv2.addWeighted(img, .5, frame, .5, 0)
     
        frame = cv2.putText(frame,name,(int(10+1/6*frame.shape[0]-text_width/2),int(offset+15+text_height/2+1/3*frame.shape[0])), font, font_scale,(20,30,0),1,cv2.LINE_AA)
    
        return frame




if __name__ == "__main__":
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
       # Windows Import
        if platform == "win32":
           # Change these variables to point to the correct folder (Release/x64 etc.) 
            sys.path.append(dir_path + '/../../../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../../../x64/Release;' +  dir_path + '/../../../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.) 
            sys.path.append('../../../../python');
           # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
           # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

   # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

   # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../../../models/"
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

    if "video" in params.keys():
        video_name=params["video"]
        if os.path.isfile(video_name):
            gui=recognitionGui(video_name)
            gui.run()
        else: 
            print("Video doesn't exist")