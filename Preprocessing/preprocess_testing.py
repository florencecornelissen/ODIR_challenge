import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image

test_image_folder = "ODIR-5K_Testing_Images"

output_path = "squared_and_cropped_dataset_test"
for file in os.listdir(output_path):
    os.remove(os.path.join(output_path, file))

def square_resized_images(input_directory, output_directory):
    for file in os.listdir(input_directory):
        image = Image.open(os.path.join(input_directory, file))
        width, height = image.size
                    
        left = 0
        top = 0
        right = width
        bottom = height
                    
        if width > height:
                # Calculate the cropping dimensions to make it square
            left = (width - height) / 2
            right = (width + height) / 2
                        
                # Only crop when image is not squared
            image = image.crop((left, top, right, bottom))

        elif height > width:
            top = int((height - width) / 2)
            bottom = int((height + width) / 2)
                        
            image = image.crop((left, top, right, bottom))
                
        image = image.resize((224, 224))
                        
                    # Save the cropped and resized images
        output_file = os.path.join(output_directory, file)
        image.save(output_file)

# Call the function to process images
square_resized_images(test_image_folder, output_path)