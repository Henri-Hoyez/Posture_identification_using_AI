from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, MaxPooling2D, UpSampling1D, Flatten,LSTM, Dense,BatchNormalization, UpSampling2D,Conv2D,concatenate
from keras.backend import squeeze
from keras.optimizers import Adam
from keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input as In
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm
from Utils import Utils

class GaitClassifier():
    def __init__(self):
        self.model = None


    def make_classifier(self,nb_cnn):
        if self.model != None:
            return

        cnn_multiple_inputs = []
        for i in range(nb_cnn):
            cnn_multiple_inputs.append(self.make_cnn())
        combined_input = concatenate([cnn.output for cnn in cnn_multiple_inputs])
        outputs = Flatten()(combined_input)
        outputs = Dense(nb_cnn,activation="relu")(outputs)
        outputs = Dense(3,activation="softmax")(outputs)

        self.model = Model([cnn.input for cnn in cnn_multiple_inputs],outputs)
        return self.model

    def make_cnn(self):
        inputs = In(shape=(32,32,3))
        for nb_filter in [128,64,32]:
            outputs = Conv2D(nb_filter,(2,2),activation="relu",padding="same")(inputs)
            outputs = BatchNormalization()(outputs)
            outputs = MaxPooling2D(2)(outputs)
        cnn = Model(inputs,outputs)
        return cnn


    def load(self,filename:str,allow_pts=None):
        data = []
        with h5py.File(filename,'r') as f:
            for keys in list(f.keys()):
                if allow_pts != None and len(allow_pts) != 0:
                    data += list(np.asarray(list(f[keys]))[:,allow_pts])
                else:
                    data += list(f[keys])
                print("blabla")
            print(np.asarray(data).shape)
        return np.asarray(data)
        

    def train(self,nb_video,nb_person,allow_pts = None, nb_cnn=25):
        data = self.load("true_spectrogram_new.h5",allow_pts)/255
        print(data.shape)
        output = np.asarray([np.asarray([i]*nb_video) for i in range(nb_person)])
        if allow_pts != None and len(allow_pts) != 0:        
            nb_cnn = len(allow_pts)
        self.make_classifier(nb_cnn)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        self.model.fit([data[:,i] for i in range(nb_cnn)], to_categorical(output.reshape(-1,1)), batch_size=64, epochs=1000, verbose=1)
        
        self.save()


    def save(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./models/classifier.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./models/classifier.h5")
        

    def test(self,nb_video,nb_person,allow_pts = None, nb_cnn=25):
        self.load_model()
        data = self.load("true_spectrogram_test_new.h5",allow_pts)/255
        print(data.shape)
        if allow_pts != None and len(allow_pts) != 0:        
            nb_cnn = len(allow_pts)
        classes = [index for index in range(nb_person)]
        output_true = np.asarray([np.asarray([i]*nb_video) for i in range(nb_person)]).reshape(-1,1)
        output_pred = np.argmax(self.model.predict([data[:,i] for i in range(nb_cnn)]),axis=1)
        print(np.abs(np.round(output_pred)).astype(int).reshape(-1))
        print("********************")
        print(output_true.reshape(-1))
        self.plot_confusion_matrix(output_true.reshape(-1),np.abs(np.round(output_pred)).astype(int).reshape(-1),np.asarray(classes))
        plt.show()
        

    def load_model(self):
        # Load model
        json_file = open('./models/classifier.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("./models/classifier.h5")

    
    def plot_confusion_matrix(self,y_true, y_pred, classes, normalize=False, title=None, cmap=cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def demo(self,pts):
        self.load_model()
        allow_pts = [8,1,0,9,10,11,12,13,14]
        utils = Utils()
        folder = "spectrogram_live"
        utils.create_spectrogram_live_test(pts,folder)
        utils.resize_data(folder)
        imgs = utils.imgs_to_tab(folder)/255
        data = imgs[allow_pts]
        preds = self.model.predict([np.asarray([data[i]]) for i in range(data.shape[0])])
        output_pred = np.argmax(preds)
        print("Predicted person :", str(output_pred))
        return output_pred, preds[0][output_pred]
            



if __name__ == "__main__":
    gait_class = GaitClassifier()
    #allow_pts = [8,1,0,9,10,11,12,13,14]
    #gait_class.train(54,3, allow_pts)
    #gait_class.test(8,3,allow_pts) 
    pts = np.random.randint(0,100,(150,75))
    gait_class.demo(pts)       