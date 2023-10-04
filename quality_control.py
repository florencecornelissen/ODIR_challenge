import pandas as pd
import os
import shutil

original_image_folder = "ODIR-5K_Training_Dataset"
csv_file = 'ODIR-5K_Training_Annotations.csv' 
copied_image_folder = "ODIR-5K_Training_Dataset_Cleaned"

# Create the copied image folder if it doesn't exist
if not os.path.exists(copied_image_folder):
    os.makedirs(copied_image_folder)

# Copy all images from the original folder to the copied folder
for image_file in os.listdir(original_image_folder):
    original_image_path = os.path.join(original_image_folder, image_file)
    copied_image_path = os.path.join(copied_image_folder, image_file)
    shutil.copyfile(original_image_path, copied_image_path)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file, delimiter=';')

# Define the search strings
search_strings = ['low image quality', 'optic disk photographically invisible', 'lens dust', 'image offset']

# Create a list to store the image file names to be deleted from the original folder
image_files_to_delete = []

# Iterate over each search string and identify image file names to delete
for search_string in search_strings:
    left_matches = df[df['Left-Diagnostic Keywords'].str.contains(search_string)]['Left-Fundus'].tolist()
    right_matches = df[df['Right-Diagnostic Keywords'].str.contains(search_string)]['Right-Fundus'].tolist()
    image_files_to_delete.extend(left_matches + right_matches)

# Remove duplicates from the list of image file names to delete
image_files_to_delete = list(set(image_files_to_delete))

# Delete the corresponding images from the original folder
for image_file in image_files_to_delete:
    print(f"Deleting: {image_file}")
    original_image_path = os.path.join(original_image_folder, image_file)
    if os.path.exists(original_image_path):
        os.remove(original_image_path)

# Save the cleaned DataFrame to a new CSV file
df = df[~(df['Left-Fundus'].isin(image_files_to_delete) | df['Right-Fundus'].isin(image_files_to_delete))]
df.to_csv("ODIR-5K_Training_Annotations_Cleaned.csv", index=False)