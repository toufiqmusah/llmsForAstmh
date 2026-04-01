# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:14:05 2026 

@author: CharlesDelahunt

This script is a modified version of 'parseAbstractContentsIntoDataframeFor2025_18april2025.py'.

Goals:
1. Parse 2025 abstracts in text file format to put into a dataframe.
    Format:
    line 1 = abstractId, ie 24-A-****-ASTMH
    line 2: Category
    line 3: title
    line 4: abstract
2. Update 2023 - 2025 category labels to match 2026 categories 

Note: abstracts range from 560 to 2800 characters. For title + abstract, the max length is 
under 3300 characters.

NOTE: This script should be run one block at a time, since there is high chance of weird one-off
crashes due to typos etc.

"""

 
import os
import numpy as np
import pandas as pd
# import codecs  # why this? Remove?
#from matplotlib import pyplot as plt 

#%% USER ENTRIES:
dataFolder = r"C:\Users\cdela\OneDrive\Documents\GitHub\llmsForAstmh\data" 

# # 2023:
# inputFile = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data" + \
#     r"\materialsForVersion_2024\astmh2023AbstractContentsCleaned_18mar2024.txt" 
# outputFilename = 'astmh_2023_AbstractContents_18april2025' 

# # 2024:    
# inputFile = os.path.join(dataFolder, r"trainingData\full2024AbstractsAsTextfile.txt") 
# outputFilename = 'astmh_2024_AbstractContents_18april2025'   

# 2023-2024 abstracts:
# This file needs (i) a few categories renamed; (ii) general category columns removed; (iii) new 
# short category names. Then the processed 2025 abstracts can be appended 
abstracts23And24 = \
    os.path.join(dataFolder, 
                 r"materialsFor2025Version\combinedAbstractContents_2023_2024_18apr2025.xlsx")

# Full 2025 abstracts:
# This text file needs to be parsed and put into a spreadsheet    
abstracts25 = os.path.join(dataFolder,
                         r"materialsFor2026Version\2025Abstracts_13mar2026.txt")
authorDataInFile = False  # This may change for new 2026 abstracts to predict.

outputSheet = 'combinedAbstractContents_2023_2024_2025_19mar2026.xlsx'
outputFilename = os.path.join(dataFolder, "materialsFor2026Version", outputSheet) 



#%% We have 3 category-specific tasks, in this order:
    # 1. Update a few 2025 names to 2026 names 
    # 2. Define alternate shorter category names, for internal use and for plots (exported results will have full names)
    # 3. Merge a few small-count categories 
    # 4. We do not assign general categories this year (since we don't have a general category model)
    
# Update category names and merges for 2026:
# This is based on 'materialsFor2026Version/ASTMH 2025 Abstract List for CDelahunt 3.13.xlsx' in repo.
categoryNameChanges2025To2026 = \
    {'Malaria - Parasite Transmission Biology':'Malaria - Parasite Biology',
     'Helminths – Nematodes – Filariasis (Clinical)':'Helminths – Nematodes – Filariasis (Other)', 
     'Helminths – Nematodes – Filariasis (Genetics/Genomics)':'Helminths – Nematodes – Filariasis (Other)',
     'Helminths – Nematodes – Filariasis (Immunology)':'Helminths – Nematodes – Filariasis (Molecular Biology and Immunology)',
     'Helminths – Nematodes – Filariasis (Epidemiology)':'Helminths – Nematodes – Filariasis (Epidemiology and Modeling)',
                       } 
# Combine a few small count categories (based on 2023-2024 data):
mergeCategoryChanges = \
    {'Kinetoplastida and Other Protozoa - Invasion, Cellular and Molecular Biology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All', 
     'Kinetoplastida and Other Protozoa - Immunology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All', 
     'Kinetoplastida and Other Protozoa - Diagnosis and New Detection Tools (Including Leishmania and Trypanosomes)':'Kinetoplastida - All', 
     'Kinetoplastida and Other Protozoa - Treatment, Drug Delivery, Drug Repurposing and Drug Discovery (Including Leishmania and Trypanosomes)':'Kinetoplastida - All', 
     'Kinetoplastida and Other Protozoa - Genomics, Proteomics and Metabolomics, Molecular Therapeutic Targets (Including Leishmania and Trypanosomes)':'Kinetoplastida - All', 
     'Kinetoplastida and Other Protozoa - Vaccines (Including Leishmania and Trypanosomes)':'Kinetoplastida - All',
     'Kinetoplastida and Other Protozoa - Epidemiology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All',
     'Kinetoplastida and Other Opportunistic and Anaerobic Protozoa - Immunology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All',
     'Kinetoplastida and Other Opportunistic and Anaerobic Protozoa - Epidemiology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All',
     'Kinetoplastida and Other Protozoa - Epidemiology (Including Leishmania and Trypanosomes)':'Kinetoplastida - All',
     'Ectoparasite-Borne Disease - Babesiosis and Lyme Disease':'Ectoparasite-Borne Disease - All', 
     'Ectoparasite-Borne Disease - Other':'Ectoparasite-Borne Disease - All',
     'Bacteriology - Trachoma':'Bacteriology - Other Bacterial Infections',
     'Helminths – Nematodes – Filariasis (Cellular and Molecular Biology)':'Helminths – Nematodes – Filariasis (Other)',
     'Helminths – Nematodes – Filariasis (Treatment and Morbidity Management)':'Helminths – Nematodes – Filariasis (Other)'}  

allCategories2026 = \
    ['Ectoparasite-Borne Disease - Babesiosis and Lyme Disease', 'Ectoparasite-Borne Disease - Other', 
     'Mosquitoes - Biology, Physiology and Immunity', 'Mosquitoes - Molecular Biology, Population Genetics and Genomics', 
     'Mosquitoes - Biology and Genetics of Insecticide Resistance', 'Mosquitoes - Bionomics, Behavior and Surveillance', 
     'Mosquitoes - Epidemiology and Vector Control', 'Arthropods/Entomology - Other', 'Bacteriology - Antimicrobial Resistance', 
     'Bacteriology - Enteric Infections', 'Bacteriology - Systemic Infections', 'Bacteriology - Trachoma', 
     'Bacteriology - Other Bacterial Infections', 'Clinical Tropical Medicine', 'HIV and Tropical Co-Infections', 
     'Global Health - Maternal, Newborn and Child Health and Nutrition', 'Global Health - Diversity, Inclusion, Decolonization and Human Rights', 
     'Global Health - Planetary Health including Climate Change', 'Global Health - Security/Emerging Infection Preparedness, Surveillance and Response(s)', 
     'Global Health - Information/Communication/Technologies Solutions in Global Health including Modeling', 'Global Health - Other', 
     'Measures for Control and Elimination of Neglected Tropical Diseases (NTDs)', 
     'One Health: The Interface of Humans, Ecosystems, and Animal Health', 'Kinetoplastida and Other Protozoa - Epidemiology (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Invasion, Cellular and Molecular Biology (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Immunology (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Diagnosis and New Detection Tools (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Treatment, Drug Delivery, Drug Repurposing and Drug Discovery (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Genomics, Proteomics and Metabolomics, Molecular Therapeutic Targets (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida and Other Protozoa - Vaccines (Including Leishmania and Trypanosomes)', 'Malaria - Pathogenesis', 
     'Malaria - Genetics, Genomics and Evolution', 'Malaria - Prevention', 'Malaria - Elimination', 'Malaria - Epidemiology', 
     'Malaria - Diagnosis - Challenges and Innovations', 'Malaria - Antimalarial Resistance and Chemotherapy', 
     'Malaria - Drug Development and Clinical Trials', 'Malaria - Immunology', 'Malaria - Parasite Biology', 
     'Malaria - Vaccines and Immunotherapeutics', 'Malaria – Surveillance and Data Utilization', 'Helminths – Nematodes – Intestinal Nematodes', 
     'Cestodes (including taeniasis and cysticercosis, echinococcosis/hydatid disease, and others)', 
     'Schistosomiasis and Other Trematodes – Immunology, Pathology, Cellular and Molecular Biology', 
     'Schistosomiasis and Other Trematodes – Epidemiology and Control', 'Schistosomiasis and Other Trematodes – Diagnostics and Treatment', 
     'Helminths – Nematodes – Filariasis (Diagnostics and Therapeutics)', 'Helminths – Nematodes – Filariasis (Molecular Biology and Immunology)', 
     'Helminths – Nematodes – Filariasis (Epidemiology and Modeling)', 'Helminths – Nematodes – Filariasis (Treatment and Morbidity Management)', 
     'Helminths – Nematodes – Filariasis (Behavioral and Social Sciences)', 'Helminths – Nematodes – Filariasis (Other)', 'Respiratory Infections',
     'Viruses - Transmission Biology', 'Viruses - Vaccine Clinical Trials', 'Viruses - Immunology', 'Viruses - Pathogenesis and Animal Models', 
     'Viruses - Evolution and Genomic Epidemiology', 'Viruses - Field and ecological studies of viruses, including surveillance and spillover risk and emergence', 
     'Viruses - Epidemiology', 'Viruses - Therapeutics and Antiviral Drugs', 'Viruses - Emerging Viral Diseases', 
     'Water, Sanitation, Hygiene and Environmental Health']

allShortCategories2026 = \
    ['Ectoparasite - Babesiosis, Lyme', 'Ectoparasite - Other', 
     'Mosquitoes - Bio, Physio, Immunity', 'Mosquitoes - Molecular Bio, Genetics', 
     'Mosquitoes - Insecticide Resistance', 'Mosquitoes - Bionomics, Behavior, Surveillance', 
     'Mosquitoes - Epidemiology, Vector Control', 'Arthropods/Entomology - Other ', 
     'Bacteriology - Antimicrobial Resistance', 'Bacteriology - Enteric Infections', 
     'Bacteriology - Systemic Infections', 'Bacteriology - Trachoma', 
     'Bacteriology - Other Infections', 'Clinical Tropical Medicine', 
     'HIV and Tropical Co-Infections', 
     'Global Health - Maternal, Newborn, Child Health, Nutrition', 'Global Health - Diversity', 
     'Global Health - Planetary Health', 'Global Health - Security, Preparedness, Surveillance', 
     'Global Health - Info/Comms/Tech, Modeling', 'Global Health - Other', 
     'Neglected Tropical Diseases Control, Elimination', 
     'One Health', 'Kinetoplastida - Epidemiology', 
     'Kinetoplastida - Invasion, Molecular Biology', 
     'Kinetoplastida - Immunology', 
     'Kinetoplastida - Diagnosis', 
     'Kinetoplastida - Treatment, Drugs', 
     'Kinetoplastida - Genomics, Molecular Therapeutic Targets', 
     'Kinetoplastida - Vaccines', 'Malaria - Pathogenesis', 
     'Malaria - Genetics', 'Malaria - Prevention', 'Malaria - Elimination', 
     'Malaria - Epidemiology', 
     'Malaria - Diagnosis', 'Malaria - Antimalarial Resistance', 
     'Malaria - Drug Dev, Trials', 'Malaria - Immunology', 'Malaria - Parasite Biology', 
     'Malaria - Vaccines', 'Malaria – Surveillance', 'Helminths – Intestinal Nematodes', 
     'Cestodes', 
     'Schisto – Immunology, Pathology, Cellular and Molecular Biology', 
     'Schisto – Epidemiology and Control', 'Schisto – Diagnostics, Treatment', 
     'Helminths – Diagnostics, Therapeutics', 'Helminths – Molecular Biology, Immunology', 
     'Helminths – Epidemiology, Modeling', 'Helminths – Treatment and Morbidity Management', 
     'Helminths – Behavioral and Social Sciences', 'Helminths – Other', 'Respiratory Infections',
     'Viruses - Transmission', 'Viruses - Vaccine Trials', 'Viruses - Immunology', 
     'Viruses - Pathogenesis, Animal Models', 
     'Viruses - Evolution, Genomic Epidemiology', 'Viruses - Field studies', 
     'Viruses - Epidemiology', 'Viruses - Therapeutics', 'Viruses - Emerging', 
     'Water, Sanitation, Hygiene']
    
mergedCategories2026 = \
    ['Ectoparasite-Borne Disease - All', 
     'Mosquitoes - Biology, Physiology and Immunity', 'Mosquitoes - Molecular Biology, Population Genetics and Genomics', 
     'Mosquitoes - Biology and Genetics of Insecticide Resistance', 'Mosquitoes - Bionomics, Behavior and Surveillance', 
     'Mosquitoes - Epidemiology and Vector Control', 'Arthropods/Entomology - Other', 'Bacteriology - Antimicrobial Resistance', 
     'Bacteriology - Enteric Infections', 'Bacteriology - Systemic Infections',  
     'Bacteriology - Other Bacterial Infections', 'Clinical Tropical Medicine', 'HIV and Tropical Co-Infections', 
     'Global Health - Maternal, Newborn and Child Health and Nutrition', 'Global Health - Diversity, Inclusion, Decolonization and Human Rights', 
     'Global Health - Planetary Health including Climate Change', 'Global Health - Security/Emerging Infection Preparedness, Surveillance and Response(s)', 
     'Global Health - Information/Communication/Technologies Solutions in Global Health including Modeling', 'Global Health - Other', 
     'Measures for Control and Elimination of Neglected Tropical Diseases (NTDs)', 
     'One Health: The Interface of Humans, Ecosystems, and Animal Health', 'Kinetoplastida and Other Protozoa - Epidemiology (Including Leishmania and Trypanosomes)', 
     'Kinetoplastida - All', 'Malaria - Pathogenesis', 
     'Malaria - Genetics, Genomics and Evolution', 'Malaria - Prevention', 'Malaria - Elimination', 'Malaria - Epidemiology', 
     'Malaria - Diagnosis - Challenges and Innovations', 'Malaria - Antimalarial Resistance and Chemotherapy', 
     'Malaria - Drug Development and Clinical Trials', 'Malaria - Immunology', 'Malaria - Parasite Biology', 
     'Malaria - Vaccines and Immunotherapeutics', 'Malaria – Surveillance and Data Utilization', 'Helminths – Nematodes – Intestinal Nematodes', 
     'Cestodes (including taeniasis and cysticercosis, echinococcosis/hydatid disease, and others)', 
     'Schistosomiasis and Other Trematodes – Immunology, Pathology, Cellular and Molecular Biology', 
     'Schistosomiasis and Other Trematodes – Epidemiology and Control', 'Schistosomiasis and Other Trematodes – Diagnostics and Treatment', 
     'Helminths – Nematodes – Filariasis (Diagnostics and Therapeutics)', 'Helminths – Nematodes – Filariasis (Molecular Biology and Immunology)', 
     'Helminths – Nematodes – Filariasis (Epidemiology and Modeling)', 'Helminths – Nematodes – Filariasis (Behavioral and Social Sciences)', 'Helminths – Nematodes – Filariasis (Other)', 
     'Respiratory Infections',
     'Viruses - Transmission Biology', 'Viruses - Vaccine Clinical Trials', 'Viruses - Immunology', 'Viruses - Pathogenesis and Animal Models', 
     'Viruses - Evolution and Genomic Epidemiology', 'Viruses - Field and ecological studies of viruses, including surveillance and spillover risk and emergence', 
     'Viruses - Epidemiology', 'Viruses - Therapeutics and Antiviral Drugs', 'Viruses - Emerging Viral Diseases', 
     'Water, Sanitation, Hygiene and Environmental Health']

# The indexing of this next list with 'mergedCategories2026' must be PERFECT:        
shortMergedCategories2026 = \
    ['Ectoparasite-Borne Disease - All', 
     'Mosquitoes - Bio, Physio, Immunity', 'Mosquitoes - Molec Bio, Genetics', 
     'Mosquitoes - Insecticide Resistance', 'Mosquitoes - Bionomics, Behavior, Surveillance', 
     'Mosquitoes - Epidemiology, Vector Control', 'Arthropods/Entomology - Other ', 
     'Bacteriology - Antimicrobial Resistance', 
     'Bacteriology - Enteric Infections', 'Bacteriology - Systemic Infections', 
     'Bacteriology - Other Infections', 'Clinical Tropical Medicine', 
     'HIV and Tropical Co-Infections', 
     'Global Health - Maternal, Newborn, Child Health, Nutrition', 'Global Health - Diversity', 
     'Global Health - Planetary Health', 'Global Health - Security, Prep, Surv', 
     'Global Health - Info/Comms/Tech, Modeling', 'Global Health - Other', 
     'NTDs Control, Elimination', 
     'One Health', 'Kinetoplastida - Epidemiology', 
     'Kinetoplastida - All', 'Malaria - Pathogenesis', 
     'Malaria - Genetics', 'Malaria - Prevention', 'Malaria - Elimination', 
     'Malaria - Epidemiology', 
     'Malaria - Diagnosis', 'Malaria - Antimalarial Resistance', 
     'Malaria - Drug Dev, Trials', 'Malaria - Immunology', 'Malaria - Parasite Biology', 
     'Malaria - Vaccines', 'Malaria – Surveillance', 'Helminths – Intestinal Nematodes', 
     'Cestodes', 
     'Schisto – Immuno, Patho, Cell and Molec Bio', 
     'Schisto – Epidemiology and Control', 'Schisto – Diagnostics, Treatment', 
     'Helminths – Diagnostics, Therapeutics', 'Helminths – Molec Bio, Immunology', 
     'Helminths – Epidemiology, Modeling',  
     'Helminths – Behavioral, Social Sciences', 'Helminths – Other', 'Respiratory Infections',
     'Viruses - Transmission', 'Viruses - Vaccine Trials', 'Viruses - Immunology', 
     'Viruses - Pathogenesis, Animal Models', 
     'Viruses - Evolution, Genomic Epidemiology', 'Viruses - Field studies', 
     'Viruses - Epidemiology', 'Viruses - Therapeutics', 'Viruses - Emerging', 
     'Water, Sanitation, Hygiene']

shortMergedCatsToClassify2026 = \
    ['Malaria - Antimalarial Resistance', 'Malaria - Diagnosis', 'Malaria - Drug Dev, Trials', 
     'Malaria - Elimination', 'Malaria - Epidemiology', 'Malaria - Genetics', 
     'Malaria - Immunology', 'Malaria - Parasite Biology', 'Malaria - Pathogenesis', 
     'Malaria - Prevention', 'Malaria - Vaccines', 'Malaria – Surveillance', 
     'Clinical Tropical Medicine', 'NTDs Control, Elimination', 'One Health', 
     'Global Health - Other', 'Viruses - Emerging', 'Viruses - Epidemiology', 
     'Viruses - Evolution, Genomic Epidemiology', 'Viruses - Field studies', 
     'Viruses - Immunology', 'Viruses - Pathogenesis, Animal Models', 
     'Viruses - Therapeutics', 'Viruses - Transmission', 'Viruses - Vaccine Trials']

#%% Parse new abstracts and put into dataframe: 
# 1. Create a dataframe to store data:
newAbstracts = pd.DataFrame({'abstractId':[], 'category':[], 'title':[], 
                             'abstractText':[]})
# Leave out authors (not in 2025 abstract data anyway). 

# 2. Read the file line by line and put category, title, id, and abstract body into dataframe.
# Then update categories to 2026, merge categories, and create shortMergedCats.

minAbstractLength = 560  # Lowest found was 570 characters, at line 7825 of 2024 file

# startLine = 0  # during debugging, we'll walk through to trip errors
file = open(abstracts25, 'r', encoding='utf-8')

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
         
        lineNum += 1
        line = file.readline()
        title = \
            line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip()
        # We have abstractId, category, and title.

        # Now find the abstract text:
        # Author listings: 
        # In 2024 and 2025 files received in 2025, there are no authors listed.
        # In 2023 files (we use the version from 2024), there are authors:
        if authorDataInFile:
            # There can be several lines of authors and affiliations, and the abstract can also be
            # several lines. Method:
            # 1. Author lines end with an integer, unless there is just one author. So:
            #    IF the next line does NOT end in an integer, then
            # 2. assume 1 line each for author and affiliation
            #    ELSE:
            # 2. The Affiliations lines start with an integer, unless there is only one author. So
            #    look for the first line with an integer.
            #    If a line starts with an integer, then for each line after it, check if it starts
            #    with an integer.
            #    After the affiliations have started, the first line that does NOT start with an
            #    integer is the first line of the abstract text.
            # 3. Then check each line after that for "-ASTMH", which is the next abstractId.
            # 4. Concatenate all the lines before the new abstractId line into abstractText.
            lineNum += 1
            line = file.readline()
            line = line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip() 
            if line[-1] not in ('1','2','3','4','5','6','7','8','9','0'):  # Case: One author
                lineNum += 1
                line = file.readline()  # this is the affiliation
                lineNum += 1
                line = file.readline()  # This is the first line of the abstract text
            else:  # Case: multiple authors
                while line[0] not in ('1','2','3','4','5','6','7','8','9','0'):
                    lineNum += 1
                    line = file.readline()
                    line = line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip()
                # We have now hit the Affiliations.
                while line[0] in ('1','2','3','4','5','6','7','8','9','0'):
                    lineNum += 1
                    line = file.readline()
                    line = line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip()
            # The current line is now the abstract
            
        else:  # case: there were no authors, so we just read the next line
            lineNum += 1    
            line = file.readline() # the abstract 
        abstractText = line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip()
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
            line = line.replace('\\t','').replace('\\n','').replace('\t','').replace('\n','').strip()
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
                 'title':[title],
                 'abstractText':[abstractText]}
            thisRow = pd.DataFrame.from_dict(d)
            newAbstracts = pd.concat((newAbstracts, thisRow),ignore_index=True)
    if len(line) == 0: 
        print(str(lineNum) + ': 0 line length. \n This could be an empty line or the end of the file. ' + \
              'Check the text file at this line number, \n' + \
                  'or in Notepad++ use Edit -> Line operations -> Remove all empty lines.')
        # Then return to the start of the main 'while'. The current 'line' is an abstractId

file.close()

# Sort the new abstracts by category (as done already for 2023, 2024):
newAbstracts = newAbstracts.sort_values(by='category')

#%% Load the previous data and combine the previous and new dataframes:
previousAbstracts = pd.read_excel(abstracts23And24)
# keep only some columns:
previousAbstracts = previousAbstracts[['abstractId', 'category', 'title', 'abstractText']]

#%% Combine with new abstracts:
allAbstracts = pd.concat((previousAbstracts, newAbstracts), ignore_index=True)

#%% Update category names, merge cats, create shortMergedCats:
    
# Update to 2026 categories:
allAbstracts['oldCategory'] = allAbstracts['category'] # for history
category2026 = []
for i in range(len(allAbstracts)):
    t = allAbstracts.loc[i, 'category']
    if t in categoryNameChanges2025To2026.keys(): 
        category2026.append(categoryNameChanges2025To2026[t])
    else:
        category2026.append(t)
allAbstracts['category'] = category2026

# Add merged categories column:
mergedCats = []
for i in range(len(allAbstracts)):
    t = allAbstracts.loc[i, 'category']
    if t in mergeCategoryChanges:
        mergedCats.append(mergeCategoryChanges[t])
    else:
        mergedCats.append(t)
allAbstracts['mergedCategory'] = mergedCats

# Add short categories column:
mergedCategories2026 = np.array(mergedCategories2026)
shortMergedCats = []
for i in range(len(allAbstracts)):
    t = allAbstracts.loc[i, 'mergedCategory']
    if t in mergedCategories2026:
        ind = np.where(mergedCategories2026 == t)[0][0]
        shortMergedCats.append(shortMergedCategories2026[ind])
    else:
        shortMergedCats.append('Not in list')
        print(f'{t} is not in the list of merged 2026 cats.')
allAbstracts['shortMergedCat'] = shortMergedCats
 
# Add a 'priority' column showing which categories are most important:
priorityCat = []
for i in range(len(allAbstracts)):
    t = allAbstracts.loc[i, 'shortMergedCat']
    if t in shortMergedCatsToClassify2026: 
        priorityCat.append(1)
    else:
        priorityCat.append(0)
allAbstracts['priorityCat'] = priorityCat     

# reorder columns:
allAbstracts = allAbstracts[['abstractId', 'oldCategory', 'category', 'mergedCategory', 
                             'priorityCat', 'shortMergedCat', 'title', 'abstractText']]

# 'allAbstracts' should now contain all data with merged, short, updated 2026 categories.

# print out counts of merged Categories:
print(allAbstracts['shortMergedCat'].value_counts())
#%% Save to excel:

allAbstracts.to_excel(outputFilename)

#%%
# # Shorten some names for convenience:
# if 'Cestodes' in mC:
#     mC = 'Cestodes'
# if "Viruses - Field and ecological studies of viruses" in mC:
#     mC = "Viruses - Field and ecological studies"
# if 'Viruses - Transmission Biology' in mC:
#     mC = 'Viruses - Pathogenesis and Animal Models'
# if 'Global Health - Information/Communication/Technologies' in mC:
#     mC = 'Global Health - Information/Communication/Technologies'
# if 'One Health: The Interconnection' in mC:
#     mC = 'One Health: The Interface of Humans, Ecosystems, and Animal Health'
# if 'Global Health - Security/Emerging Infection Preparedness' in mC:
#     mC = 'Global Health - Security/Preparedness'
# if 'Global Health - Planetary Health' in mC:
#     mC = 'Global Health - Other'
# if 'Schistosomiasis and Other Trematodes' in mC:
#     mC = 'Schistosomiasis and Other Trematodes'
# if 'Helminths'in mC and 'Intestinal Nematodes' in mC:
#     mC = 'Helminths - Nematodes - Intestinal Nematodes'
# if 'Helminths' in mC and 'Nematodes' in mC and 'Filariasis' in mC:            
#     mC = 'Helminths - Nematodes - Filariasis'
# if 'Bacteriology - Systemic Infections' in mC:
#     mC = 'Bacteriology - Other Bacterial Infections'
# # if 'Malaria - Parasite Transmission Biology' in mC:
# #     mC = 'Malaria - Pathogenesis and parasite transmission'
# # if 'Malaria - Pathogenesis' in mC:
# #     mC = 'Malaria - Pathogenesis and parasite transmission'
