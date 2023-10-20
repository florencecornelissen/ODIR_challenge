import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image

# read in data
df = pd.read_csv('ODIR-5K_Training_Annotations.csv', sep = ";")

# classify keywords with labels 
def classify_keywords(keywords, df_row):
    classified_diseases = []
    if 'normal' in keywords.lower():
        classified_diseases.append('N')  
    if ('diabetes' in keywords.lower() or 'diabetic' in keywords.lower() or 'proliferative retinopathy' in keywords.lower()) and df_row['D'] == 1:
        classified_diseases.append('D')  
    if 'glaucoma' in keywords.lower() and df_row['G'] == 1:
        classified_diseases.append('G')
    if 'cataract' in keywords.lower() and df_row['C'] == 1:
        classified_diseases.append('C')
    if ('amd' in keywords.lower() or 'age-related macular degeneration' in keywords.lower()) and df_row['A'] == 1:
        classified_diseases.append('A')
    if ('hypertension' in keywords.lower() or 'hypertensive' in keywords.lower()) and df_row['H'] == 1:
        classified_diseases.append('H')
    if ('myopia' in keywords.lower() or 'myopic' in keywords.lower()) and df_row['M'] == 1:
        classified_diseases.append('M')    
    if any(name in keywords.lower() for name in disease_names):
        classified_diseases.append('O')
    if not classified_diseases:
        classified_diseases.append('O')
    return classified_diseases

disease_names = [
    'laser', 'drusen', 'pigment', 'epiretinal membrane',
    'maculopathy', 'vitreous degeneration', 'myelinated nerve fibers',
    'refractive media opacity', 'tessellated fundus', 'atrophy',
    'spotted membranous change', 'occlusion', 'syndrome',
    'neovascularization', 'sheathing', 'coloboma', 'edema'
]

#classify the keywords per eye
df['class-left'] = df.apply(lambda row: classify_keywords(row['Left-Diagnostic Keywords'], row), axis=1)
df['class-right'] = df.apply(lambda row: classify_keywords(row['Right-Diagnostic Keywords'], row), axis=1)

#duplicate rows so every row contains one of the eyes
df1 = pd.DataFrame(np.repeat(df.values, 2, axis=0))
df1.columns = df.columns

def append_extension(fn, side):
    return f"{fn}_{side}.jpg"

# Create a new column 'Side' based on the index
df1['Side'] = df1.index % 2

# Apply the appropriate extension based on the 'Side' column
df1['ID'] = df1.apply(lambda row: append_extension(row['ID'], 'left' if row['Side'] == 0 else 'right'), axis=1)
df1['labels'] = np.where(df1['Side'] == 0, df1['class-left'], df1['class-right'])
df1['Diagnostic Keywords'] = np.where(df1['Side'] == 0, df1['Left-Diagnostic Keywords'], df1['Right-Diagnostic Keywords'])

#keep only the following columns
df1 = df1[['ID', 'labels', 'Diagnostic Keywords']]

original_image_folder = "ODIR-5K_Training_Dataset"
copied_image_folder = "ODIR-5K_Training_Dataset_Cleaned_wo_control"

# Create the copied image folder if it doesn't exist
if not os.path.exists(copied_image_folder):
    os.makedirs(copied_image_folder)

# Copy all images from the original folder to the copied folder
for image_file in os.listdir(original_image_folder):
    original_image_path = os.path.join(original_image_folder, image_file)
    copied_image_path = os.path.join(copied_image_folder, image_file)
    shutil.copyfile(original_image_path, copied_image_path)

# # Define the search strings
# search_strings = ['low image quality', 'optic disk photographically invisible', 'lens dust', 'image offset']

# # Create a list to store the image file names to be deleted from the copied folder
# image_files_to_delete = []

# # Iterate over each search string and identify image file names to delete
# for search_string in search_strings:
#     matches = df1[df1['Diagnostic Keywords'].str.contains(search_string)]['ID'].tolist()
#     image_files_to_delete.extend(matches)

# # Remove duplicates from the list of image file names to delete
# image_files_to_delete = list(set(image_files_to_delete))

# # Delete the corresponding images from the copied folder
# for image_file in image_files_to_delete:
#     #print(f"Deleting: {image_file}")
#     copied_image_path = os.path.join(copied_image_folder, image_file)
#     if os.path.exists(copied_image_path):
#         os.remove(copied_image_path)

# # Save the cleaned DataFrame to a new CSV file
# df1 = df1[~(df1['ID'].isin(image_files_to_delete))]
# df1.to_csv("ODIR-5K_Training_Preprocessed.csv", index=False)

# dataset_path = "ODIR-5K_Training_Dataset_Cleaned"
output_path = "squared_and_cropped_dataset_wo_control"
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
square_resized_images(copied_image_folder, output_path)
