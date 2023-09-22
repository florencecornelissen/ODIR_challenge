import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from collections import Counter

csv_file = 'ODIR-5K_Training_Annotations.csv' 
df = pd.read_csv(csv_file, delimiter=';')

def classify_keywords(keywords):
    classified_diseases = []
    if 'normal' in keywords.lower():
        classified_diseases.append('N')  
    if 'diabetes' in keywords.lower() or 'diabetic' in keywords.lower() or 'retinopathy' in keywords.lower():
        classified_diseases.append('D')  
    if 'glaucoma' in keywords.lower():
        classified_diseases.append('G')
    if 'cataract' in keywords.lower():
        classified_diseases.append('C')
    if 'amd' in keywords.lower() or 'age-related macular degeneration' in keywords.lower():
        classified_diseases.append('A')
    if 'hypertension' in keywords.lower() or 'hypertensive' in keywords.lower():
        classified_diseases.append('H')
    if 'myopia' in keywords.lower():
        classified_diseases.append('M')    
    if not classified_diseases:
        classified_diseases.append('O')
    return classified_diseases

df['class-left'] = df['Left-Diagnostic Keywords'].apply(classify_keywords)
df['class-right'] = df['Right-Diagnostic Keywords'].apply(classify_keywords)

# Get counts original, left, right
disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
df['Diseases'] = df[disease_columns].apply(lambda row: [col for col in disease_columns if row[col] == 1], axis=1)

# Export csv
df.to_csv('ODIR-5K_Training_Annotations_LR.csv', index=False)

# Visualise counts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df['class-left'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Counts of class-left')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
df['class-right'].value_counts().plot(kind='bar', ax=axes[1])
axes[1].set_title('Counts of class-right')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
df['Diseases'].value_counts().plot(kind='bar', ax=axes[2])
axes[2].set_title('Counts of Diseases')
axes[2].set_xlabel('Disease')
axes[2].set_ylabel('Count')
plt.tight_layout()
plt.show()

#what keywords are not used yet?
# unknown_other_df = df[(df['class-left'] == 'other') & (df['class-right'] == 'other')]
# for index, row in unknown_other_df.iterrows():
#     print(f"Index: {index}")
#     print(f"Left Keywords: {row['Left-Diagnostic Keywords']}")
#     print(f"Right Keywords: {row['Right-Diagnostic Keywords']}")
#     print('-' * 50)
