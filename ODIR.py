import imageio as iio
from PIL import Image
import numpy as np
import os
import sys
import argparse
import pickle

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

# Another method, which says that pictures should not be beyond 800x600 (to train much quicker). However, all our pictures are. https://towardsdatascience.com/deep-learning-prepare-image-for-dataset-d20d0c4e30de 
def rescale_images(directory, size):
  for img in os.listdir(directory):
    im = Image.open(directory+img)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(directory+img)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Rescale images")
  parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
  parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
  args = parser.parse_args()
  rescale_images(args.directory, args.size)
##run this command
##python convert_image_resolution.py -d images/ -s 800 600