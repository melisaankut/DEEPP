# -*- coding: utf-8 -*-
#In the case of using TPU, much faster results will be obtained.
#However, there is a problem in loading the hickle file due to 
#the python version. TPU can be used together with the file 
#reading parts in the comment line.
"""import os
use_tpu = True #@param {type:"boolean"}

if use_tpu:
    assert 'COLAB_TPU_ADDR' in os.environ, 'Missing TPU; did you request a TPU in Notebook Settings?'

if 'COLAB_TPU_ADDR' in os.environ:
  TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
else:
  TF_MASTER=''"""

"""# TPU address
tpu_address = TF_MASTER
epochs = 50
steps_per_epoch = 5

import tensorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TF_MASTER)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
"""

import pandas as pd
import numpy as np
from skimage.io import imread
import cv2 as cv
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score
import itertools
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

"""df = pd.read_csv('all_data_info.csv')
df
new = df.loc[np.logical_and(np.logical_or(df['artist_group'] == 'train_only',df['artist_group'] == 'train_and_test'), df['in_train'] == True )]
new = new.rename(columns = {'style': 'painting_style'}, inplace = False)
new.to_csv('filtered_data',index = True)
item_counts = new["painting_style"].value_counts()
print(item_counts.head(10))
print(item_counts[0:10].index.tolist())
print(new.shape)

dict_class = {}
for i in item_counts[0:10].index.tolist():
  dict_class[i] = 0
"""

#Since our dataset is 36GB, we kept the train_img and labels numpy array we created 
#with the '.hkl' extension instead of doing image reading and elimination every time.
#You do not have to execute the parts taken in the comment line.
"""
train_img = []
labels = []
def load_image_files(image_path):
    file = image_path
    count = 0
    img = imread(str(file))
    img_pred = cv.resize(img, (256, 256))
    if (img_pred.shape == (256,256,3)):
      train_img.append(img_pred)
    else:
      return False

counter = 0
for i in new['new_filename']:
  y = new.iloc[counter]['painting_style']
  if y==y and (y in dict_class):
    if (dict_class[y] > 10):
      pass
    else:
      try:
        f_name = '/content/drive/MyDrive/kaggle/train/' + i
        result = load_image_files(f_name)
        if result != False:
          labels.append(y)
          dict_class[y] += 1
          if dict_class.values.count(11) == len(dict_class.values):
            break
          #print(f_name , y ,counter , type(y))
      except:
        pass
  counter += 1
  if counter % 10 == 0:
    print(counter)
"""

"""keys_list = list(set(labels))
values_list = []
for i in range(len(set(labels))):
  values_list.append(i)
print(values_list)
zip_iterator = zip(keys_list, values_list)
label_dictionary = dict(zip_iterator)

label_int_list = []
for key in labels:
  label_int_list.append(label_dictionary[key])

print(label_int_list)

data = train_img
hkl.dump( train_img, 'train_img.hkl' )
hkl.dump( label_int_list, 'labels.hkl' )"
"""

train_img_hkl = hkl.load('/path/train_img/hickle/file') #upload hickle file from directory

label_hkl = hkl.load('/path/labels/hickle/file')        #upload hickle file from directory

label_count = 10
train_img_hkl = np.array(train_img_hkl)
y = np.array(label_hkl)
# split the dataset
X_train = train_img_hkl[:16000]
X_test = train_img_hkl[16000:]
y_train = y[:16000]
y_test = y[16000:]

import os
import tensorflow as tf

#Convolutional autoencoder implementation
input_img = keras.Input(shape=(256, 256, 3))
x = layers.Conv2D(100, (5, 5), activation='relu', padding='same')(input_img)   
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(200, (5, 5), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.UpSampling2D((2, 2))(encoded)
x = layers.Conv2D(200, (5, 5), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(100, (5, 5), activation='relu', padding='same')(x)
decoded = layers.Conv2D(3, (5, 5), activation="sigmoid", padding="same")(x)
optimizer = tf.train.AdamOptimizer(0.05)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer=optimizer, loss='categorical_crossentropy')
"""
with strategy.scope():
  autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),   // compile for TPU
                loss=tf.keras.losses.sparse_categorical_crossentropy, 
                metrics=['accuracy'])
autoencoder.summary()
"""

print(X_train.shape)
print(X_test.shape)

autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test)
                )

decoded_imgs_train = autoencoder.predict(X_train)
decoded_imgs_test = autoencoder.predict(X_test)

n_classes = label_count
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(256, 256, 3)))
# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(label_count, activation='softmax'))
optimizer = keras.optimizers.Adam()
optimizer.learning_rate.assign(0.05)
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

# training the model for 40 epochs
mdl = model.fit(decoded_imgs_train, Y_train, batch_size = 32, epochs=40 , validation_data=(X_test, Y_test))

y_pred = []
y_pred_matrix = model.predict(X_test)
for i in y_pred_matrix:
  y_pred.append(np.argmax(i))
y_pred = np.array(y_pred)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (80,10)
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):

   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print('Normalized confusion matrix')
   else:
     print('Confusion matrix, without normalization')
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, ['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)', 'Baroque', 'Surrealism', 'Symbolism', 'Rococo'] ,rotation='vertical')
   plt.yticks(tick_marks, ['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)', 'Baroque', 'Surrealism', 'Symbolism', 'Rococo'])
 
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
 
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label') 

confusion_matrix= confusion_matrix(y_test,y_pred)
plot_confusion_matrix(confusion_matrix, list(set(y_test + y_pred)),normalize=True)
report = metrics.classification_report(y_test, y_pred, target_names= ['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 'Post-Impressionism', 'Art Nouveau (Modern)', 'Baroque', 'Surrealism', 'Symbolism', 'Rococo'])
print(report)