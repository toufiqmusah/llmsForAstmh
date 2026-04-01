# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:14:19 2024

@author: CharlesDelahunt

Goal: test keyword feature function

"""

import os
import numpy as np
import pandas as pd

from astmhSupportFunctions_6april2024 import populateKeywordFeatureVector_fn

dataFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data"

# Load the keywords into a list:
with open(os.path.join(dataFolder, 'keywords_8april2024.txt'), 'r') as file:
    line = file.readlines()  # only one line
keywords = line[0].split(',')

data = pd.read_excel(os.path.join(dataFolder, 'astmh2023AbstractContents_26mar2024.xlsx'))

abstract = data.loc[9, 'abstractText']
keywordFeatureVector = populateKeywordFeatureVector_fn(abstract, keywords, [0, 1])
