import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import model_from_json

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

def test_model():

	# Load model
	json_file = open('./models/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("./models/model.h5")

	test = pd.read_csv("./test/dataset_test.csv",sep=";",names=['path','class_name','id_class','keypoints'])
	test_data_input = list(test.keypoints.values)
	test_data_output = list(test.id_class.values)
	for i in range(0,len(test_data_input)):
	    tmp = test_data_input[i]
	    tmp = tmp.split(",")
	    test_data_input[i] = np.asarray([float(elem) for elem in tmp])


	predictions = model.predict(np.asarray(test_data_input))

	for i in range(0,len(test_data_output)):
	    if test_data_output[i]==0:
	        test_data_output[i]=[1,0,0]
	    elif test_data_output[i]==1:
	        test_data_output[i]=[0,1,0]
	    elif test_data_output[i]==2:
	        test_data_output[i]=[0,0,1]


	y_init = []
	y_predition = []
	for j in range(int(len(predictions))):
	    y_init.append(np.argmax(np.asarray(test_data_output)[j]))
	    y_predition.append(np.argmax(np.asarray(predictions)[j]))

	plot_confusion_matrix(y_init, y_predition, classes=['laying','sitting','standing'], title='Confusion matrix, without normalization')
	plt.show()

if __name__ == "__main__":
	test_model()