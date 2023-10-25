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
from keras.applications import ResNet50, VGG19, VGG16
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from keras.models import load_model
import pickle
import keras_tuner as kt

X_train = np.load('./Datasets/X_train.npy')
X_val = np.load('./Datasets/X_val.npy')
y_train = np.load('./Datasets/y_train.npy')
y_val = np.load('./Datasets/y_val.npy')
with open('./Datasets/class_weights.pkl', 'rb') as file:
    loaded_class_weights = pickle.load(file)

print('--------------------------- VGG16 kerasTuner -------------------------------')

IMG_SHAPE = X_train[0].shape


# Define the hyperparameters
batch_size = [16, 32, 64, 128, 256]
epochs = [20]
learn_rate = [0.001]
dropout = [0.2]

tf.random.set_seed(42)
print('Did set random seet 42 in this run')

def build_model(hp):
                    hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3 ,0.4])
                    hp_units = hp.Int('units', min_value=64, max_value=512, step=32)
                    base_model = VGG16(include_top = False, weights = 'imagenet', input_shape = IMG_SHAPE)
                    base_model.trainable = False
                    model= Sequential()
                    model.add(InputLayer(input_shape = IMG_SHAPE))
                    model.add(base_model)
                    model.add(Flatten())
                    model.add(Dense(hp_units,activation=('relu'))) 
                    model.add(Dropout(hp_dropout))
                    model.add(Dense(8,activation=('sigmoid')))

                    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

                    adam = Adam(learning_rate=hp_learning_rate)

                    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['AUC', 'accuracy'])

                    return model

def append_extension(fn, side):
        return f"{fn}_{side}.jpg"


best_val_accuracy = 0.0
best_model = None
best_batch = 0
best_epoch = 0
best_learn = 0
best_dropout = 0

# Model is saved at the end of every epoch, if it's the best seen so far.
checkpoint_directory = f'./CheckpointVGG16_kerasTuner/'

if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_filepath = './CheckpointVGG16_kerasTuner'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',  
    patience=2,              
    restore_best_weights=True  
)


tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=20, factor=3, directory='VGG16tunerdirectory', project_name='ODIR_challenge')

tuner.search(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[model_checkpoint_callback, early_stopping_callback])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search for VGG16 is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')}, the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}, adn the optimal dropout is {best_hps.get('dropout')}.
""")

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

model = load_model('./CheckpointVGG16_kerasTuner/')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])

batch_size=2**6
y_test = model.predict(X_test, steps=len(testdf)/batch_size)

for i,j in enumerate(['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']):
  testdf[j]=y_test[:,i]

# Weer de columns omzetten naar de juiste class volgorde om het in te leveren
testdf = testdf[['ID', 'ID number', 'N' ,'D' ,'G', 'C', 'A' ,'H' ,'M', 'O']]

# For each two rows, take the max of the two rows
testdf = testdf.groupby(testdf.index // 2).max()

testdf = testdf.drop(columns=['ID number'])

testdf.to_csv('submissionVGG16_kerasTuner.csv', index=False)

