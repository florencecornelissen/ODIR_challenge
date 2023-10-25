from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Rescaling, InputLayer
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from IPython.display import Image, display
import pandas as pd
import numpy as np
from keras.optimizers import RMSprop, Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import re
import tensorflow as tf
from keras import backend as K
from keras.applications import ResNet50, VGG19, DenseNet201, NASNetMobile, VGG16, MobileNetV2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from keras.models import load_model
import pickle
from tensorflow.keras import regularizers

print('Final combined testing NASNetMobile 200 epochs - patience 20')

X_train = np.load('./Datasets/X_train.npy')
X_val = np.load('./Datasets/X_val.npy')
y_train = np.load('./Datasets/y_train.npy')
y_val = np.load('./Datasets/y_val.npy')
with open('./Datasets/class_weights.pkl', 'rb') as file:
    loaded_class_weights = pickle.load(file)

IMG_SHAPE = X_train[0].shape

print('This is with rescaling')
#Initializing the hyperparameters
batch_size=256
initial_epochs=200
learn_rate=0.0001
dropout=0.2
hidden_units=128


def build_model(dropout=dropout, hidden_units=hidden_units, learn_rate=learn_rate):
    base_model = NASNetMobile(include_top = False, weights = 'imagenet', input_shape = IMG_SHAPE)
    base_model.trainable = False
    model= Sequential()
    model.add(InputLayer(input_shape = IMG_SHAPE))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(hidden_units,activation=('relu')))
    model.add(Dropout(dropout))
    model.add(Dense(8,activation=('sigmoid')))

    adam = Adam(learning_rate=learn_rate)

    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['AUC', 'accuracy'])

    return model

model = build_model()

# Model is saved at the end of every epoch, if it's the best seen so far.
checkpoint_directory = './CheckpointNASNetMobile_combined200epochs/'

if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_filepath = './CheckpointNASNetMobile_combined200epochs'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',  
    patience=20,              
    restore_best_weights=True  
)

class CustomDataGenerator(Sequence):
    def __init__(self, X, y, batch_size, class_weights):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.class_weights = class_weights

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return batch_X, batch_y

    def on_epoch_end(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        pass

train_generator = CustomDataGenerator(X_train, y_train, batch_size, loaded_class_weights)

history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=(X_val,y_val),
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_steps=len(X_val),
                    callbacks=[model_checkpoint_callback, early_stopping_callback],
                    verbose = 1)

# Plotting

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy NASNetMobile finetuned')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
# plt.ylim([0,1.0])
plt.title('Training and Validation Loss NASNetMobile finetuned')
plt.xlabel('epoch')
plt.savefig('finetuned_NASNetMobile200epochs.png')


### Testing set ###

testdfstart= pd.read_csv('XYZ_ODIR.csv')

testdf = pd.DataFrame(np.repeat(testdfstart.values, 2, axis=0))
testdf.columns = testdfstart.columns

def append_extension(fn, side):
    return f"{fn}_{side}.jpg"

testdf['Side'] = testdf.index % 2

testdf['ID number'] = testdf.loc[:, 'ID']

testdf['ID number'] = testdf.apply(lambda row: append_extension(row['ID number'], 'left' if row['Side'] == 0 else 'right'), axis=1)

testdf.drop(columns=['Side'], inplace=True)

testdf = testdf[['ID', 'ID number', 'A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']]

test_dir = './squared_and_cropped_dataset_test/'
test_img = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
img_size = 224

# Form the test labels
X_test = []
for image in tqdm(test_img):
    try:
        img = cv2.imread(image)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            X_test.append(img)
    except:
        continue

X_test = np.asarray(X_test, dtype=np.float32)

model = load_model(checkpoint_directory)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])

# batch_size=2**6
y_test = model.predict(X_test, steps=len(testdf)/batch_size)

for i,j in enumerate(['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']):
  testdf[j]=y_test[:,i]

# Weer de columns omzetten naar de juiste class volgorde om het in te leveren
testdf = testdf[['ID', 'ID number', 'N' ,'D' ,'G', 'C', 'A' ,'H' ,'M', 'O']]

# For each two rows, take the max of the two rows
testdf = testdf.groupby(testdf.index // 2).max()

testdf = testdf.drop(columns=['ID number'])

testdf.to_csv('submissionNASNetMobile_final200epochs.csv', index=False)