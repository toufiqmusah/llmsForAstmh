# -*- coding: utf-8 -*-
"""
Created sometime in April 2024

@author: Olivia Zahn?

Goal: Generate k-fold split on a chosen column.

"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

dataFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data"
inputFile = "combinedAbstractContents_2023_2024_18apr2025.xlsx"  # Training data   
# Used for the 2024 version: "astmh2023AbstractContents_26mar2024.xlsx", now in data/materiasForVersion_2024
date = 'merged_18apr2025'

# %%
# Read Excel file
df = pd.read_excel(os.path.join(dataFolder, inputFile))
# Define features and labels
X = df[['abstractId','generalCategory','title', 'abstractText', 'mergedCategory', 'category']]
y = df[['mergedCategory']]


# %%
# Initialize 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# %%
# Generate cross-validation splits
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    train_by_split = pd.concat([X_train, y_train], axis=1)
    test_by_split = pd.concat([X_test, y_test], axis=1)

    # Save train and test splits as new Excel files
    train_filename = f'train_split_merged_{fold}_' + date + '.xlsx'
    test_filename = f'test_split_merged_{fold}_' + date + '.xlsx'
    train_by_split.to_excel(dataFolder + "/" + train_filename, index=True, header=True)
    test_by_split.to_excel(dataFolder + "/" + test_filename, index=True, header=True)
    
    print(f"Fold {fold}:")
    print(f"Train Set saved to {train_filename} - X: {len(X_train)}, y: {len(y_train)}")
    print(f"Test Set saved to {test_filename} - X: {len(X_test)}, y: {len(y_test)}")
    print("="*30)