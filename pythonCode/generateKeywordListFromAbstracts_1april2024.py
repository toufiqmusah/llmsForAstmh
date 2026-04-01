# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:22:53 2024

@author: CharlesDelahunt

Goal: Generate a list of keywords (species, etc) from Abstracts.

"""

import os
import numpy as np
import pandas as pd

#%%
dataFile = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data" + \
    r"\astmh2023AbstractContents_26mar2024.xlsx"

d = pd.read_excel(dataFile, sheet_name='Sheet1')

allTitles = []
for i in d['title'].values:
    allTitles = allTitles + '___' + i

allAbstracts = ''
for i in d['abstractText'].values:
    allAbstracts = allAbstracts + ' \n' + i

keywords = []
for i in range(65, 91):  # ascii for capital letters. space has ascii = 32
    print(chr(i) + '_')
    ab = allAbstracts
    while len(ab) > 20:
        n = ab.find(chr(i) + '_')  # one value
        if n != None:
            ab = ab[n:]
            m = np.array(ab.find(' '))
            t = np.array(ab.find('-'))
            if t != None:
                m = min(t, m)
            this = ab[:m]
            this = this.replace(',','').replace('.','')
            if this not in keywords:
                keywords.append(this)
            ab = ab[m:]
