import imageio as iio
import numpy as np
import os
import sys

cwd = os.getcwd() 

# Reading an image
img = iio.imread("./ODIR-5K_Training_Dataset/0_left.jpg")
print(img)
# iio.imwrite("g4g.jpg", img)

images = []

directory = './ODIR-5K_Training_Dataset'
for item in os.listdir(directory):
    if item.endswith(".jpg"):
        file = os.path.join(directory, item)
        picture = iio.imread(file)
        images.append(picture)