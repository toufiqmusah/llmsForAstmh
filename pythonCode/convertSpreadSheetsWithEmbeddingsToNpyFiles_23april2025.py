# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:58:51 2025

@author: CharlesDelahunt

Goal: create an .npy of embedding vectors for a given spreadsheet that has a column with 
those embeddings.
"""
import os
import pandas as pd
import numpy as np

#%% USER ENTRIES:
folder = r"V:\FAMLI\Results\Olivia\astmh"

# filename = "embedded_test_split_merged_1_18apr2025.xlsx"
# saveFilename = "embeddings_test_split_merged_1.npy"

filename = "embedded_train_split_merged_1_18apr2025.xlsx"
saveFilename = "embeddings_train_split_merged_1.npy"

# filename = "embedded_newAbstractsToReclassify_merged_18apr2025.xlsx"
# saveFilename = "embeddings_2025_merged.npy"

#%% Extract each feature vector. These are strings, with values separated by single or double
# spaces:

d = pd.read_excel(os.path.join(folder, filename))    
v = d['abstractText_embedding'].values
numFeatures = 768
 
embeddings = -1 * np.ones((len(v), numFeatures))
for i in range(len(v)):
    s = v[i]
    s = s.replace('  ', ' ').replace('[ ', '').replace('[','').replace(']','')
    s = s.split(' ')
    if len(s) == numFeatures:
        for j in range(numFeatures):
            embeddings[i, j] = float(s[j])
    else:
        print(str(i) + 'th row does not split properly')

#%%
np.save(saveFilename, embeddings)        