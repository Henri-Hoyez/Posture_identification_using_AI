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





data_dim = 75
timesteps = 30
num_classes = 3

list_of_datasets=[]
for root, dirs, files in os.walk("./datasets"):
    for file in files:
        list_of_datasets.append(os.path.join(root,file))


input_train = []
output_train = []
count = 0

for dataset in list_of_datasets:
    #if count == 5:
    #    break
    count += 1
    data_train = pd.read_csv(dataset,sep=";",names=['path','class_name','id_class','keypoints'])
    length = data_train.id_class.count()

    print(dataset)

    for i in range(timesteps,length):
        data_tmp = data_train.values[i-timesteps:i,-1]
        tmp=[]
        for d in range(timesteps):
            tmp.append(np.array([float(elem) for elem in data_tmp[d].split(",")]).flatten())
        input_train.append(tmp)
        output_train.append(data_train.id_class[i])

for i in range(len(output_train)):
    res = [0,0,0]
    res[output_train[i]] = 1
    output_train[i] = res

print(np.asarray(input_train).shape)
print(np.asarray(output_train).shape)

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(np.asarray(input_train), np.asarray(output_train),
          batch_size=64, epochs=30, verbose=1)

print(np.asarray(input_train)[0,:,:])
print(np.argmax(np.asarray(output_train)[0]))
print(np.argmax(model.predict(np.asarray([np.asarray(input_train)[0,:,:]]))))



predictions = model.predict(np.asarray(input_train))

print("Prédiction terminée..")


# serialize model to JSON
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./model.h5")
print("Saved model to disk")

output_train = np.asarray(output_train)
predictions = np.asarray(predictions)

y_init = []
y_predition = []
bar = tqdm(total=len(predictions))
for j in range(len(predictions)):
    y_init.append(np.argmax(output_train[j]))
    y_predition.append(np.argmax(predictions[j]))
    bar.update(1)
bar.close()
plot_confusion_matrix(y_init, y_predition, classes=['boxing','jumping','walking'], title='Confusion matrix, without normalization')
plt.show()
