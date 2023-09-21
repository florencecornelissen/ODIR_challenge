import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

csv_file = 'ODIR-5K_Training_Annotations.csv' 
df = pd.read_csv(csv_file, delimiter=';')

def classify_keywords(keywords):
    if 'normal' in keywords.lower():
        return 'normal'    
    elif 'cataract' in keywords.lower():
        return 'cataract'
    elif 'amd' in keywords.lower() or 'age-related macular degeneration' in keywords.lower():
        return 'AMD'
    elif 'hypertension' in keywords.lower() or 'hypertensive' in keywords.lower():
        return 'hypertension'
    elif 'myopia' in keywords.lower():
        return 'myopia'
    elif 'glaucoma' in keywords.lower():
        return 'glaucoma'
    elif 'diabetes' in keywords.lower() or 'diabetic' in keywords.lower() or 'retinopathy' in keywords.lower():
        return 'diabetes'
    else:
        return 'other'

df['class-left'] = df['Left-Diagnostic Keywords'].apply(classify_keywords)
df['class-right'] = df['Right-Diagnostic Keywords'].apply(classify_keywords)
df.to_csv('ODIR-5K_Training_Annotations_LR.csv', index=False)

#plot disease counts: original, left, and right
diseases = ["N", "D", "G", "C", "A", "H", "M", "O"]
disease_labels = ["normal", "diabetes", "glaucoma", "cataract", "AMD", "hypertension", "myopia", "other"]
df['original_class'] = 'unknown'
for disease, label in zip(diseases, disease_labels):
    df['original_class'] = np.where(df[disease] == 1, label, df['original_class'])
left_class_counts = df['class-left'].value_counts()
right_class_counts = df['class-right'].value_counts()
original_class_counts = df['original_class'].value_counts()

# plt.figure(figsize=(18, 5))
# plt.subplot(1, 3, 1)
# original_class_counts.plot(kind='bar', color='lightgreen')
# plt.title('Original Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.subplot(1, 3, 2)
# left_class_counts.plot(kind='bar', color='skyblue')
# plt.title('Class Distribution - Left Eye')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.subplot(1, 3, 3)
# right_class_counts.plot(kind='bar', color='lightcoral')
# plt.title('Class Distribution - Right Eye')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

#plot original disease counts (including combinations)
binary_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'O', 'M']
binary_combinations = list(itertools.combinations(binary_columns, 2))
combination_counts = {}
for combination in binary_combinations:
    combination_name = ' and '.join(combination)
    count = ((df[list(combination)] == 1).all(axis=1)).sum()
    if count > 0:  
        combination_counts[combination_name] = count
single_variable_counts = {}
for column in binary_columns:
    count = (df[column] == 1).sum()
    if count > 0:  
        single_variable_counts[column] = count        
all_counts = {**combination_counts, **single_variable_counts}

# plt.figure(figsize=(12, 8))
# plt.bar(all_counts.keys(), all_counts.values())
# plt.xlabel('Variable Combinations')
# plt.ylabel('Count of 1s')
# plt.title('Counts of Binary Variable Combinations and Single Variables Being Equal to 1')
# plt.xticks(rotation=90)
# plt.show()

#what keywords are not used yet?
unknown_other_df = df[(df['class-left'] == 'other') & (df['class-right'] == 'other')]
# for index, row in unknown_other_df.iterrows():
#     print(f"Index: {index}")
#     print(f"Left Keywords: {row['Left-Diagnostic Keywords']}")
#     print(f"Right Keywords: {row['Right-Diagnostic Keywords']}")
#     print('-' * 50)

