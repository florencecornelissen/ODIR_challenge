import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import re
import torch
import ast
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

df = pd.read_csv("ODIR-5K_Training_Annotations.csv")

df = df[['ID', 'labels']]
df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x))
dfdummy = pd.get_dummies(df['labels'].apply(pd.Series).stack()).sum(level=0)
df = pd.concat([df, dfdummy], axis=1)

# Change every item in labels to classes from 0 to 7
df['labels'] = df['labels'].apply(lambda x: [0 if i == 'A' else 1 if i == 'C' else 2 if i == 'D' else 3 if i == 'G' else 4 if i == 'H' else 5 if i == 'M' else 6 if i == 'N' else 7 for i in x])

# Convert the labels from A, C,D, G, H, M, n and O to a column of lists
df['coded labels'] = df.apply(lambda x: [x['A'], x['C'], x['D'], x['G'], x['H'], x['M'], x['N'], x['O']], axis=1)

df['Name'] = df['ID'].apply(lambda x: os.path.splitext(x)[0])
df['Name'] = df['Name'].str.cat(df['coded labels'].astype(str), sep ="_")

output_path = "./squared_and_cropped_dataset_wo_control/"

for item in range(len(df['ID'])):
    os.rename(output_path + df['ID'][item], output_path + df['Name'][item] + '.jpg')

train_img = [os.path.join(output_path, i) for i in os.listdir(output_path)]

def createxy(image_list):
    X = []  # images
    y = []  # labels (0 for Normal or 1 for Pneumonia)
    pattern = r'\[([\d,\s]+)\]'

    for image in tqdm(image_list):
        try:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img = cv2.normalize(img, img , 0, 255, cv2.NORM_MINMAX)
            # img = np.clip(img, 0 , 255).astype(np.float32)
            X.append(img)
        except:
            continue

        match = re.search(pattern, image)
        extracted_list = match.group(1).split(',')
        extracted_list = [int(i) for i in extracted_list]
        y.append(extracted_list)

    return X, y
    
X_train, y_train = createxy(train_img)

# Define your augmentations
augmentations = A.Compose([
    A.Rotate(limit=15, p=0.5),               # Random rotation up to 15 degrees
    A.HorizontalFlip(p=0.5),                 # Horizontal flipping with a 50% probability
    A.RandomBrightnessContrast(p=0.5),       # Random brightness and contrast adjustments
    A.Blur(blur_limit=(1, 3), p=0.5),        # Gaussian blur
], p=1)


def augmented_set(X_train, y_train, augmentations):
    X_train_augmented = []
    y_train_augmented = []
    for i in range(len(X_train)):
        # get image and label
        image = X_train[i]
        label = y_train[i]

        # augment image
        augmented = augmentations(image=image)
        augmented_image = augmented['image']
        
        # append image and label
        X_train_augmented.append(augmented_image)
        y_train_augmented.append(label)

    return X_train_augmented, y_train_augmented

num_classes = len(y_train[0])  # Get the number of classes

# Initialize lists for each class
class_X_train = [[] for _ in range(num_classes)]
class_y_train = [[] for _ in range(num_classes)]

# Loop through the data and assign each sample to the appropriate class
for i in range(len(X_train)):
    labels = y_train[i]
    for j in range(num_classes):
        if labels[j] == 1:
            class_X_train[j].append(X_train[i])
            class_y_train[j].append(labels)

# Fix class imbalance using downsampling and upsampling with augmentation
augmented_data = {class_index: {'X': [], 'y': []} for class_index in range(num_classes)}
target_samples_per_class = 500

classes_to_downsample = [class_index for class_index in range(num_classes) if len(class_X_train[class_index]) > target_samples_per_class]
classes_to_upsample = [class_index for class_index in range(num_classes) if len(class_X_train[class_index]) < target_samples_per_class]

for class_index in range(num_classes):
    x, y = class_X_train[class_index], class_y_train[class_index]    
    if class_index in classes_to_downsample:
        sampled_indices = random.sample(range(len(x)), target_samples_per_class)
        x_downsampled = [x[i] for i in sampled_indices]
        y_downsampled = [y[i] for i in sampled_indices]
        augmented_data[class_index]['X'] = x_downsampled
        augmented_data[class_index]['y'] = y_downsampled
    else:        
        samples_to_generate = target_samples_per_class - len(x)
        x_augmented, y_augmented = augmented_set(x, y, augmentations)[:samples_to_generate]
        augmented_data[class_index]['X'] = x_augmented
        augmented_data[class_index]['y'] = y_augmented

for class_index in classes_to_downsample:
    class_X_train[class_index] = augmented_data[class_index]['X']
    class_y_train[class_index] = augmented_data[class_index]['y']
    
for class_index in classes_to_upsample:
    class_X_train[class_index].extend(x_augmented)
    class_y_train[class_index].extend(y_augmented)

# Print the sizes of each class after augmentation
for class_index in range(num_classes):
    x_size = len(class_X_train[class_index])
    y_size = len(class_y_train[class_index])
    print(f"Class {class_index}: X_train size = {x_size}, y_train size = {y_size}")

# Convert to one list
class_X_train = sum(class_X_train, [])
class_y_train = sum(class_y_train, [])

X_train1, X_validation1, y_train1, y_validation1 = train_test_split(class_X_train, class_y_train, test_size=(500/4146), random_state=42, shuffle=True)

# Define the number of classes
num_classes = 8

# Define the number of samples in each class
class_sizes = [548, 581, 500, 594, 461, 536, 500, 500]

# Calculate the total number of samples
total_samples = np.sum(class_sizes)

# Calculate class weights
class_weights = {i: total_samples / (num_classes * class_sizes[i]) for i in range(num_classes)}

# Print the class weights
for i, weight in class_weights.items():
    print(f"Class {i}: Weight = {weight:.4f}")

class_weights = dict(sorted(class_weights.items()))
class_weights

with open('./Datasets_wo_control/class_weights.pkl', 'wb') as file:
    pickle.dump(class_weights, file)

np.save('./Datasets/X_train.npy', X_train1)
np.save('./Datasets/y_train.npy', y_train1)
np.save('./Datasets/X_val.npy', X_validation1)
np.save('./Datasets/y_val.npy', y_validation1)

