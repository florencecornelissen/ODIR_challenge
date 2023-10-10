import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from collections import Counter

# Data
csv_file = 'ODIR-5K_Training_Annotations_Cleaned.csv' 
df = pd.read_csv(csv_file, delimiter=',')

# Labels
def classify_keywords(keywords, df_row):
    classified_diseases = []
    if 'normal' in keywords.lower():# and df_row['N'] == 1:
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
        print(keywords.lower())
    return classified_diseases

disease_names = [
    'laser', 'drusen', 'pigment', 'epiretinal membrane',
    'maculopathy', 'vitreous degeneration', 'myelinated nerve fibers',
    'refractive media opacity', 'tessellated fundus', 'atrophy',
    'spotted membranous change', 'occlusion', 'syndrome',
    'neovascularization', 'sheathing', 'coloboma', 'edema'
]

df['class-left'] = df.apply(lambda row: classify_keywords(row['Left-Diagnostic Keywords'], row), axis=1)
df['class-right'] = df.apply(lambda row: classify_keywords(row['Right-Diagnostic Keywords'], row), axis=1)

disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
disease_counts = df[disease_columns].sum()

# Plot
color_mapping = {
    'N': '#1f77b4',
    'D': '#ff7f0e',
    'G': '#2ca02c',
    'C': '#d62728',
    'A': '#9467bd',
    'H': '#8c564b',
    'M': '#e377c2',
    'O': '#7f7f7f'
}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
left_counts = df['class-left'].explode().value_counts().sort_values(ascending=False)
left_counts.plot(kind='bar', ax=axes[0], color=[color_mapping.get(label, 'gray') for label in left_counts.index], width=1)
axes[0].set_title('Counts of Labels Left')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
right_counts = df['class-right'].explode().value_counts().sort_values(ascending=False)
right_counts.plot(kind='bar', ax=axes[1], color=[color_mapping.get(label, 'gray') for label in right_counts.index], width=1)
axes[1].set_title('Counts of Labels Right')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
disease_counts = df[disease_columns].sum().sort_values(ascending=False)
disease_counts.plot(kind='bar', ax=axes[2], color=[color_mapping.get(label, 'gray') for label in disease_counts.index], width=1)
axes[2].set_title('Counts of Labels Original')
axes[2].set_xlabel('Disease')
axes[2].set_ylabel('Count')
colors = sns.color_palette("pastel")
sns.set_palette(colors)
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.subplots_adjust(wspace=0.2) 
plt.show()

# Export csv
df.to_csv('ODIR-5K_Training_Annotations_Cleaned_LabelsLR.csv', index=False)

