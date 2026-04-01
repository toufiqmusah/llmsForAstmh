# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:54:20 2024

@author: CharlesDelahunt

2024 version

Goal: parse the text file version of the ASTMH 2024 Abstracts by line, and populate a pandas
dataframe.
Different (and easier) than 2023 abstracts:
    No authors or affiliations
    No general category
The abstract can be spread over multiple lines.

"""
import os
import numpy as np
import pandas as pd
import codecs  # why this? Remove?
#from matplotlib import pyplot as plt  # matplotlib is broken in env1 on home laptop (18 mar 2024)

#%% User entries
inputFile = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data" + \
    r"\astmh2024AbstractContentsAsTextFile_22april2024.txt"

df = pd.DataFrame({'abstractId':[], 'category':[], 'mergedCategory':[],
                   'generalCategory':[], 'shortGenCat':[], 'title':[],
                   'abstractText':[]})
# Leave out authors.
outputFilename = 'astmh2024AbstractContents_22april2024'  # we'll save as excel

minAbstractLength = 560  # Lowest found was 568 characters, at line 3149

genCat = ['Malaria', 'Viruses', 'Global Health', 'Mosquitoes', 'Bacteriology',
           'Kinetoplastida and Other Protozoa', 'Helminths-Nematodes',
           'Clinical Tropical Medicine', 'Pneumonia, Respiratory Infections and Tuberculosis',
           'Schistosomiasis and Other Trematodes', 'One Health',
           'Integrated Control Measures for Neglected Tropical Diseases (NTDs)',
           'Water, Sanitation, Hygiene and Environmental Health', 'Ectoparasite-Borne Disease',
           'Cestodes', 'Arthropods/Entomology', 'HIV and Tropical Co-Infections']
shortCat = ['Malaria', 'Viruses', 'Global Health', 'Mosquitoes', 'Bacteriology',
               'Kinetoplastida', 'Helminths', 'Clinical Trop Med', 'Pneumonia TB',
               'Schistosomiasis', 'One Health', 'Integrated Control', 'Water Sanitation',
               'Ectoparasite-Borne', 'Cestodes', 'Arthropods', 'HIV']
shortCatDict = {genCat[i]: shortCat[i] for i in range(len(genCat))}

#%% Read the file line by line:
# startLine = 0  # during debugging, we'll walk through to trip errors
file = open(inputFile, 'r', encoding='utf-8')

lineNum = 0
# # If starting at a later line, ignore the first many lines:
# if startLine > 0:
#     while lineNum < startLine:
#         lineNum += 1
#         a = file.readline()

# Since our while statement assumes an existing abstractId, get the first line, which is an
# abstractId:
lineNum += 1
line = file.readline()
if "-ASTMH" not in line:
    print(str(lineNum) + ': Error... line is not an abstractId')
while len(line) > 0:
    # Update to console:
    if (lineNum - 1) % 100 == 0:
        print(str(lineNum), end='   ')

    if "-ASTMH" in line: # This identifies the abstractId
        abstractId = line.replace('\\t','').replace('\\n','').strip()
        lineNum += 1
        line = file.readline()
        category = line.replace('\\t','').replace('\\n','').strip()
        # For mergedCategory, apply some bespoke merges to get rid of very small
        # categories:
        mC = category
        if 'Kinetoplastida' in mC:
            mC = 'Kinetoplastida - All'
        if 'Ectoparasite-Borne Disease' in mC:
            mC = 'Ectoparasite-Borne Disease - All'
        if 'Trachoma' in mC:
            mC = 'Bacteriology - Other Bacterial Infections'
        if 'Helminths' in mC and 'Diagnostics and Therapeutics' in mC:
            mC = 'Helminths - Nematodes - Filariasis (Other)'
        # Shorten some names for convenience:
        if 'Cestodes' in mC:
            mC = 'Cestodes'
        if "Viruses - Field and ecological studies of viruses" in mC:
            mC = "Viruses - Field and ecological studies of viruses"
        if 'Global Health - Information/Communication/Technologies' in mC:
            mC = 'Global Health - Information/Communication/Technologies'
        if 'One Health: The Interconnection' in mC:
            mC = 'One Health: Interconnections'
        if 'Global Health - Security/Emerging Infection Preparedness' in mC:
            mC = 'Global Health - Security/Preparedness'
        if 'Schistosomiasis and Other Trematodes - Immunology' in mC:
            mC = 'Schistosomiasis and Other Trematodes - Immunology Etc'
        mergedCategory = mC  # rename for clarity
        # For general category, first some fussing to make sure that hypens are in
        # the right place:
        temp = category.replace(' – ',' - ')  # get rid of \u2013 type hyphen
        temp = temp.split(' - ')[0]
        # Here are the short general categories:
        shortGenCatList = \
            ('Arthropods','Bacteriology', 'Cestodes', 'Clinical Trop Med', 'Ectoparasite-Borne',
             'Global Health', 'HIV', 'Helminths', 'Integrated Control', 'Kinetoplastida',
             'Malaria', 'Mosquitoes', 'One Health', 'Pneumonia TB', 'Schistosomiasis',
             'Viruses', 'Water Sanitation')
        for i in shortGenCatList:
            if i in temp:
                shortGenCat = i
                if i == 'Clinical Trop Med':
                    if 'Clinical Tropical' in temp:
                        shortGenCat = i
                if i == 'Integrated Control':
                    if 'Control and Elimination' in temp:
                        shortGenCat = i

        lineNum += 1
        line = file.readline()
        title = line.replace('\\t','').replace('\\n','').strip()
        # We have abstractId, category, and title.
        # Abstract is next:
        # Then check each line after this for "-ASTMH", which is the next abstractId.
        # Concatenate all the lines before the new abstractId line into abstractText.
        lineNum += 1
        line = file.readline()
        abstractText = line.replace('\\t','').replace('\\n','')
        # # Replace species abbreviations with _ to connect the species name. The exceptions
        # # have been manually given an extra space. So we look for " A. " but not " A.  ":
        # asciiCapCodes = range(65,91)
        # for i in asciiCapCodes:
        #     s = ' ' + chr(i) + '. '
        #     t = ' ' + chr(i) + '_'
        #     sRevert = ' ' + chr(i) + '_ '
        #     tRevert = ' ' + chr(i) + '. '
        #     abstractText = abstractText.replace(s, t).replace(sRevert, tRevert)
        # Keep reading lines, looking for the clue that the next abstractId has arrived:
        while ("-ASTMH" not in line) and (len(line) > 0):
            lineNum += 1
            line = file.readline()
            line = line.replace('\\t','').replace('\\n','')
            if (not line) or ("-ASTMH" not in line):   # Detect end of file
                abstractText = abstractText + ' ' + line
        # We have now hit the next 'abstractId'.
        # Check that 'abstractText' is long enough (error check), then add it to the dataframe.
        if len(abstractText) < minAbstractLength:
            print('Line ' + str(lineNum) + ': Abstract is too short.')
            break
        else:
            abstractText = abstractText.strip()
            d = {'abstractId':[abstractId],
                 'category':[category],
                 'mergedCategory':[mergedCategory],
                 'generalCategory':[shortGenCat],
                 'shortGenCat':[shortGenCat],
                 'title':[title],
                 'abstractText':[abstractText]}
            thisRow = pd.DataFrame.from_dict(d)
            df = pd.concat((df, thisRow),ignore_index=True)
    if len(line) == 0:
         print(str(lineNum) + ': 0 line length.')
        # Then return to the start of the main 'while'. The current 'line' is an abstractId
#%%
file.close()
#%% Save to excel:

df.to_excel(outputFilename + '.xlsx')
