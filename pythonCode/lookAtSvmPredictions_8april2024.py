# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:06:24 2024

@author: CharlesDelahunt

Goal: examine Ishan's FPs in crucial classes for keywords.
"""

import os
import numpy as np
import pandas as pd

svmResultsFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\results\ishan_results"
trainPredictions = 'train_svm_predictions.csv'
testPredictions = 'test_svm_predictions.csv'

dataFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data"
mainData = 'astmh2023AbstractContents_26mar2024.xlsx'

d = pd.read_excel(os.path.join(dataFolder, mainData))

trainP = pd.read_csv(os.path.join(svmResultsFolder, trainPredictions))
testP = pd.read_csv(os.path.join(svmResultsFolder, testPredictions))
svmDf = pd.concat((trainP, testP))

# Add a column to 'd' with svm predictions:
svmPreds = np.array(['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'] * len(d))
mainAbIds = d['abstractId'].values
svmAbId = svmDf['abstractId'].values
svmGuess = svmDf['svm_preds'] .values
for i in range(len(svmAbId)):
    this = svmAbId[i]
    ind = np.where(mainAbIds == this)[0]
    if len(ind) == 0:
        print(str(i) + ' ' + this + ' has no match.')
    else:
        svmPreds[ind] = svmGuess[i]
# smvPreds = [i.replace('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '--') for i in svmPreds]
d.insert(loc=6,column='svmPrediction', value=svmPreds)


#%% Get inds of FPs in important categories:
genCat = d['shortGenCat'].values
svmPred = d['svmPrediction'].values
abstracts = d['abstractText']

cat = ['Malaria', 'Global Health', 'Integrated Control']
fpList = []
for c in cat:
    print(c + ' FPs:')
    for i in range(len(genCat)):
        if genCat[i] != c and svmPred[i] == c:
            fpList.append(i)
    print('/n')

    # with open(c + '_fpAbstracts.txt', 'w') as file:

                # print(str(i), end=', ')
                # file.write('\n')
                # file.write(genCat[i] + ', FP in ' + c + ':')
                # file.write(abstracts[i])
   # file.close()

fpData = d.loc[fpList, :]
fpData.to_excel('svmFPsForImportantCategories_8april2024.xlsx')
