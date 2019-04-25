#!/usr/bin/env python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import random
from test_model import test_model

num_classes = 3

datasets_names_list=[]
for root, dirs, files in os.walk("./datasets"):
    for file in files: 
        datasets_names_list.append(os.path.join(root,file))

train = pd.concat(map(lambda x: pd.read_csv(x,names=['path','class_name','id_class','keypoints'],sep=";"), datasets_names_list))
train_data_input = list(train.keypoints.values)
train_data_output = list(train.id_class.values)

for i in range(0,len(train_data_input)):
    tmp = train_data_input[i]
    tmp = tmp.split(",")
    train_data_input[i] = np.asarray([float(elem) for elem in tmp])

model = Sequential()
model.add(Dense(128, input_shape=(75,),activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

tmp = [i for i in range(0,len(train_data_output))]
random.shuffle(tmp)
train_data_input=[train_data_input[i] for i in tmp]
train_data_output=[train_data_output[i] for i in tmp]

print(np.asarray(train_data_input).shape)
print(np.asarray(train_data_output).shape)

for i in range(0,len(train_data_output)):
    if train_data_output[i]==0:
        train_data_output[i]=[1,0,0]
    elif train_data_output[i]==1:
        train_data_output[i]=[0,1,0]
    elif train_data_output[i]==2:
        train_data_output[i]=[0,0,1]

model.fit(np.asarray(train_data_input),np.asarray(train_data_output),epochs=100, verbose=2, batch_size=256)

# serialize model to JSON
model_json = model.to_json()
with open("./models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./models/model.h5")
print("Saved model to disk")

test_model()