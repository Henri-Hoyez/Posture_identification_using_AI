import numpy as np
import time
import classes
from display_box import getBoundingBox
from keras.models import model_from_json
from queue import Queue
from angles_from_body25 import Geometry_Body25
import requete

class PredictionService(object):

    instance = None

    def __new__(self, name, model):
        if not PredictionService.instance:
            if name == "pose":
                PredictionService.instance = PredictionService.__Posture(model)
            elif name == "action":
                PredictionService.instance = PredictionService.__Action(model)
            elif name == "gait":
                PredictionService.instance = PredictionService.__Gait(model)
            elif name == "webcam":
                PredictionService.instance = None
        return PredictionService.instance

    # Load model
    def loadModel(name):
        json_file = open('./models/'+name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('./models/'+name+'.h5')
        return model

    class __Posture():

        def __init__(self,model):
            self.model = PredictionService.loadModel(model)
            self.dataset = []
        

        def getData(self):
            return self.dataset

        def displayPrediction(self, frame, kp):
            
            self.dataset = []
            normalized_keypoints = np.array(kp)
            if len(normalized_keypoints.shape) == 0:
                return frame
            for index in range(normalized_keypoints.shape[0]):
                #Normalize keypoints
                normalized_keypoints[index,:,:] = self.normalize(normalized_keypoints[index,:,:])

                # Check that the necessary points are detected
                isDetected = (kp[index,11,:].sum() != 0)
                isDetected = (isDetected and (kp[index,14,:].sum() != 0))
                isDetected = (isDetected and (kp[index,8,:].sum() != 0))
                isDetected = (isDetected and (kp[index,1,:].sum() != 0))
                # Keep in memory the history of images
                if isDetected and normalized_keypoints.shape[0] < 4:
                    new_data = ",".join(map(str,list(np.asarray(normalized_keypoints[index,:,:].flatten()))))
                    self.dataset.append(new_data)

                # Predict the pose
                predictions = self.model.predict(np.asarray([normalized_keypoints[index,:,:].flatten()]))
                # Display the prediction
                frame = getBoundingBox(frame,kp,index,classes.getPoseFromId(np.argmax(predictions)),predictions[0][np.argmax(predictions)], isDetected)
            return frame

        def normalize(self, kp):
            min_X = min(kp[:,0])
            max_X = max(kp[:,0])
            min_Y = min(kp[:,1])
            max_Y = max(kp[:,1])
            min_Z = min(kp[:,2])
            max_Z = max(kp[:,2])
            kp[:,0] = (kp[:,0]-min_X)/(max_X-min_X)
            kp[:,1] = (kp[:,1]-min_Y)/(max_Y-min_Y)
            kp[:,2] = (kp[:,2]-min_Z)/(max_Z-min_Z)
            return kp

    class __Action():

        def __init__(self,model):
           self.model = PredictionService.loadModel(model)
           self.keypoints_queue = Queue(maxsize = 20)
           self.tv_on = False
           self.command_delay = time.time()

        def displayPrediction(self, frame, kp):

            normalized_keypoints = np.array(kp)
            if len(normalized_keypoints.shape) == 0 :
                while not self.keypoints_queue.empty():
                    self.keypoints_queue.get()
                return frame

            normalized_keypoints[0,:,:] = self.normalize(normalized_keypoints[0,:,:])

            if self.keypoints_queue.full() :
                self.keypoints_queue.get()
            self.keypoints_queue.put(np.asarray(normalized_keypoints)[0,:,:].flatten())

            predictions = np.array([])
            if self.keypoints_queue.full() :
                _input = np.asarray([list(self.keypoints_queue.queue)])
                predictions = self.model.predict(_input)
                if classes.getActionFromId(np.argmax(predictions)) == "clapping" and time.time()-self.command_delay >= 10:
                    requete.requete(4 if not self.tv_on else 5)
                    self.tv_on = not self.tv_on
                    self.command_delay = time.time()

            if predictions.size == 0:
                frame = getBoundingBox(frame,kp,0,classes.getActionFromId(6),-1,True)
            else:
                frame = getBoundingBox(frame,kp,0,classes.getActionFromId(np.argmax(predictions)),predictions[0][np.argmax(predictions)],True)
            return frame


        def normalize(self, kp):
            min_X = min(kp[:,0])
            max_X = max(kp[:,0])
            min_Y = min(kp[:,1])
            max_Y = max(kp[:,1])
            min_Z = min(kp[:,2])
            max_Z = max(kp[:,2])
            kp[:,0] = (kp[:,0]-min_X)/(max_X-min_X)
            kp[:,1] = (kp[:,1]-min_Y)/(max_Y-min_Y)
            kp[:,2] = (kp[:,2]-min_Z)/(max_Z-min_Z)
            return kp

    class __Gait():

        def __init__(self,model):
           self.model = PredictionService.loadModel(model)
           self.keypoints_queue=Queue(maxsize=15)
           self.person_queue=Queue(maxsize=5)
           self.geometry = Geometry_Body25()

        def displayPrediction(self, frame, kp):

            if len(kp.shape) == 0:
                while not self.keypoints_queue.empty():
                    self.keypoints_queue.get()
                while not self.person_queue.empty():
                    self.person_queue.get()
                return frame

            if self.keypoints_queue.full():
                self.keypoints_queue.get()
            angles = self.geometry.getAngles(kp[0,:,:])
            distances = self.geometry.getDistances(kp[0,:,:])
            self.keypoints_queue.put(np.append(angles,distances))
            
            _preds=np.array([])
            if self.keypoints_queue.full():
                _input=np.asarray([list(self.keypoints_queue.queue)])
                predictions=self.model.predict(_input)

                if self.person_queue.full():
                    self.person_queue.get()
                self.person_queue.put(predictions)

                if self.person_queue.full():
                    _preds=np.asarray([list(self.person_queue.queue)])
                    _preds=np.sum(_preds,axis=0)[0]/self.person_queue.qsize()

            if _preds.size == 0:
                frame= getBoundingBox(frame,kp,0,classes.getGaitFromId(5),-1,True)
            else:
                frame= getBoundingBox(frame,kp,0,classes.getGaitFromId(np.argmax(_preds)),_preds[0][np.argmax(_preds)],True)
            return frame