from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
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





data_dim = 10+4
timesteps = 15

list_of_datasets=[]
directories=[]
for root, dirs, files in os.walk("datasets_generated"):
    for file in files:
        directories.append(dirs)
        list_of_datasets.append(os.path.join(root,file))


input_train = []
output_train = []

classes=[]

for dataset in list_of_datasets:
    data_train = pd.read_csv(dataset,sep=";",names=['path','class_name','id_class','angles','distances'])
    length = data_train.id_class.count()

    try:
        if not data_train.class_name.values[0] in classes:
            classes.append(data_train.class_name.values[0])
            print(data_train.class_name.values[0]) 
    except:
        pass
    for i in range(timesteps,length):
        angles = data_train.values[i-timesteps:i,-2]
        distances = data_train.values[i-timesteps:i,-1]
        tmp=[]
        for d in range(timesteps):
            tab_a = np.array([float(elem) for elem in angles[d].split(",")])
            tab_d = np.array([float(elem) for elem in distances[d].split(",")])
            tmp.append(np.append(tab_a,tab_d))
        input_train.append(tmp)
        output_train.append(data_train.id_class.values[i])

num_classes = len(classes)
print(classes)
for i in range(len(output_train)):
    res = [0] * num_classes
    res[output_train[i]] = 1
    output_train[i] = res

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(128, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(128, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


print(len(input_train))
print(len(output_train))

ipt = np.asarray(input_train)
opt = np.asarray(output_train)

model.fit(ipt, opt, batch_size=256, epochs=30, verbose=1)


# serialize model to JSON
model_json = model.to_json()
with open("./models/model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./models/model3.h5")
print("Saved model to disk")