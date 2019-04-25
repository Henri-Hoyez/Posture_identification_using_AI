import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import model_from_json
import os
from tqdm import tqdm
import argparse 

data_dim = 10+4
timesteps = 15

def getId(name):
  if "Baptiste" in name:
    return 0
  elif "Henri" in name:
    return 1
  elif "Jerome" in name:
    return 2
  elif "Louis" in name:
    return 3
  elif "Lucas" in name:
    return 4
  elif "Nicolas" in name:
    return 5
  elif "Zoe" in name:
    return 6

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
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def test_model(norm=True):

  # Load model
  json_file = open('./models/model3.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights("./models/model3.h5")

  list_of_datasets=[]
  for root, dirs, files in os.walk("./datasets_generated_test"):
      for file in files:
          list_of_datasets.append(os.path.join(root,file))


  input_test = []
  output_test = []

  classes=[]

  for dataset in list_of_datasets:

      data_test = pd.read_csv(dataset,sep=";", names=['path','class_name','id_class','angles','distances'])
      length = data_test.id_class.count()

      try:
          if not data_test.class_name.values[0] in classes:
              classes.append(data_test.class_name.values[0])
              print(data_test.class_name.values[0]) 
      except:
          pass

      for i in range(timesteps,length):
          angles = data_test.values[i-timesteps:i,-2]
          distances = data_test.values[i-timesteps:i,-1]
          tmp=[]
          for d in range(timesteps):
            tab_a = np.array([float(elem) for elem in angles[d].split(",")])
            tab_d = np.array([float(elem) for elem in distances[d].split(",")])
            tmp.append(np.append(tab_a,tab_d))
          input_test.append(tmp)
          output_test.append(getId(data_test.class_name[i]))

  num_classes = len(classes)

  for i in range(len(output_test)):
      res = [0] * num_classes
      res[output_test[i]] = 1
      output_test[i] = res


  predictions = model.predict(np.asarray(input_test))
  print("Prédiction terminée..")

  output_test = np.asarray(output_test)
  predictions = np.asarray(predictions)

  y_init = []
  y_predition = []
  bar = tqdm(total=len(predictions))
  for j in range(len(predictions)):
      y_init.append(np.argmax(output_test[j]))
      y_predition.append(np.argmax(predictions[j]))
      bar.update(1)
  bar.close()

  plot_confusion_matrix(y_init, y_predition, classes=classes,normalize=norm, title='Confusion matrix (test), with normalization')
  plt.savefig("confusion_matrix_test.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--norm", action="store_true", help="Normalize data")
  args = parser.parse_args()
  test_model(True)