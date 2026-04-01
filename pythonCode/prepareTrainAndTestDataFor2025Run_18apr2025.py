# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:36:06 2025

@author: CharlesDelahunt

Goal: update the category and umbrella labels for 2023 and 2024 abstracts, to make a training set
for the 2025 model.

"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc

rc('font', weight='bold')

#%%
dataFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data"
d23 = pd.read_excel(os.path.join(dataFolder, "astmh_2023_AbstractContents_18april2025.xlsx"))
d24 = pd.read_excel(os.path.join(dataFolder, "astmh_2024_AbstractContents_18april2025.xlsx"))
# Note: the version of 2024 abstracts dated 22apr2024 has only 772 out of 2433 (it was just 
# the subset to be reclassified at the time)

# We need to fix a few typos in the 2025 batch:
d25 = pd.read_excel(os.path.join(dataFolder, 
                                 "astmh_2025_AbstractContentsToReclassify_18april2025.xlsx"))

# These are the corrected categories for 2023 and 2024, if any, per abstract:
c24 = pd.read_excel(os.path.join(dataFolder,
              r"dataIncludingRelabeledPriorYearsFor2025_17april2025\trainingData",
              "2024 abstracts to CDelahunt 3.28.xlsx"))
c23 = pd.read_excel(os.path.join(dataFolder,
              r"dataIncludingRelabeledPriorYearsFor2025_17april2025\trainingData",
              "2023 abstracts to CDelahunt 3.28.xlsx"))

c23 = c23.fillna('')
c24 = c24.fillna('')

# Fix a couple typos in c23 and c24:
bad = 'One Health: : The Interface of Humans, Ecosystems and Animal Health'
good = 'One Health: The Interface of Humans, Ecosystems, and Animal Health'
c23.loc[c23['NEW NAME FOR CATEGORY IN 2025'] == bad,'NEW NAME FOR CATEGORY IN 2025'] = good
c24.loc[c24['NEW NAME FOR CATEGORY IN 2025'] == bad,'NEW NAME FOR CATEGORY IN 2025'] = good
# Align abstract identifiers:
newIds = ['23-A-' + str(i) + '-ASTMH' for i in c23['CONTROLNUMBER']]
c23.insert(loc=1, column='abstractId', value=newIds)

newIds = ['24-A-' + str(i) + '-ASTMH' for i in c24['CONTROLNUMBER']]
c24.insert(loc=1, column='abstractId', value=newIds)

d23.insert(loc=2, column='oldCategory',value=d23['category'].values)
d24.insert(loc=2, column='oldCategory',value=d24['category'].values)

#%% Update categories to 2025 versions:
# 1. For 2023: Update categories if either col D or E has an entry
# 2. For 23 and 24: Update categories to 2025 versions
# 3. Along the way, update the category label for 2024 abstracts, since the labels we have are from
# before reassigning happened

# 2023:
print('processing 2023 data...')
theseAbstractIds = np.array(d23['abstractId'].values)
for i in range(len(c23)):
    rowInd = np.where(theseAbstractIds == c23.loc[i, 'abstractId'])[0]
    if len(rowInd) > 0:
        rowInd = rowInd[0]
        if c23.loc[i, 'TOPIC1'] != d23.loc[rowInd, 'category']:
            d23.loc[rowInd, 'category'] = c23.loc[i, 'TOPIC1'] 
        if c23.loc[i, 'NEW NAME FOR CATEGORY IN 2024'] != '':
            d23.loc[rowInd, 'category'] = c23.loc[i, 'NEW NAME FOR CATEGORY IN 2024']
        if c23.loc[i, 'NEW NAME FOR CATEGORY IN 2025'] != '':
            d23.loc[rowInd, 'category'] = c23.loc[i, 'NEW NAME FOR CATEGORY IN 2025']
       
    else:
        print(str(i) + ' ' + c23.loc[i, 'abstractId'] + \
              ': no matching prior abstract id found.')

# 2024:
print('processing 2024 data...')
theseAbstractIds = np.array(d24['abstractId'].values)
for i in range(len(c24)):
    rowInd = np.where(theseAbstractIds == c24.loc[i, 'abstractId'])[0]
    if len(rowInd) > 0:
        rowInd = rowInd[0]
        if c24.loc[i, 'TOPIC1'] != d24.loc[rowInd, 'category']:
            d24.loc[rowInd, 'category'] = c24.loc[i, 'TOPIC1']
        if c24.loc[i, 'NEW NAME FOR CATEGORY IN 2025'] != '':
            d24.loc[rowInd, 'category'] = c24.loc[i, 'NEW NAME FOR CATEGORY IN 2025']
    else:
        print(str(i) + ' ' + c24.loc[i, 'abstractId'] + \
              ': no matching prior abstract id found.')

#%% Combine d23 and d24 to get a full training set:
trainSet23_24 = pd.concat([d23, d24], ignore_index=True)

trainSet23_24.to_excel('combinedAbstractContents_2023_2024_18apr2025.xlsx')

#%% Correct a few non-existent category names in d25 (make best guess as to what they meant), then
# resave:
    
bad = 'Malaria - Transmission Biology'
good = 'Malaria - Parsite Transmission Biology'
d25.loc[d25['category'] == bad, ('category', 'mergedCategory')] = good

bad = 'Malaria - Biology and Pathogenesis'
good = 'Malaria - Pathogenesis'
d25.loc[d25['category'] == bad, ('category', 'mergedCategory')] = good

bad = 'Malaria - Prevention and Elimination Strategies'
good = 'Malaria - Elimination'
d25.loc[d25['category'] == bad, ('category', 'mergedCategory')] = good

bad = 'Malaria - Parsite Transmission Biology'
good = 'Malaria - Parasite Transmission Biology'
d25.loc[d25['category'] == bad, ('category', 'mergedCategory')] = good

bad = 'Malaria - Epidemiology, Surveillance and Data Utilization'
good = 'Malaria - Surveillance and Data Utilization'
d25.loc[d25['category'] == bad, ('category', 'mergedCategory')] = good

d25.to_excel(os.path.join(dataFolder, 
             "astmh_2025_AbstractContentsToReclassify_withCorrectedTypos_18april2025.xlsx"))
             
#%% The official 2025 categories:
cat25 = ['Ectoparasite-Borne Disease - Babesiosis and Lyme Disease', 'Ectoparasite-Borne Disease - Other', 'Arthropods/Entomology - Other', 'Mosquitoes - Biology and Genetics of Insecticide Resistance', 'Mosquitoes - Biology, Physiology and Immunity', 'Mosquitoes - Bionomics, Behavior and Surveillance', 'Mosquitoes - Epidemiology and Vector Control', 'Mosquitoes - Molecular Biology, Population Genetics and Genomics', 'Bacteriology - Antimicrobial Resistance', 'Bacteriology - Enteric Infections', 'Bacteriology - Other Bacterial Infections', 'Bacteriology - Systemic Infections', 'Bacteriology - Trachoma', 'Clinical Tropical Medicine', 'HIV and Tropical Co-Infections', 'Global Health - Maternal, Newborn and Child Health and Nutrition', 'Global Health - Diversity, Inclusion, Decolonization and Human Rights', 'Global Health - Information/Communication/Technologies Solutions in Global Health including Modeling', 'Global Health - Other', 'Global Health - Planetary Health including Climate Change', 'Global Health - Security/Emerging Infection Preparedness, Surveillance and Response(s)', 'Measures for Control and Elimination of Neglected Tropical Diseases (NTDs)', 'One Health: The Interface of Humans, Ecosystems, and Animal Health', 'Kinetoplastida and Other Protozoa - Diagnosis and New Detection Tools (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Epidemiology (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Genomics, Proteomics and Metabolomics, Molecular Therapeutic Targets (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Immunology (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Invasion, Cellular and Molecular Biology (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Treatment, Drug Delivery, Drug Repurposing and Drug Discovery (Including Leishmania and Trypanosomes)', 'Kinetoplastida and Other Protozoa - Vaccines (Including Leishmania and Trypanosomes)', 'Malaria - Antimalarial Resistance and Chemotherapy', 'Malaria - Diagnosis - Challenges and Innovations', 'Malaria - Drug Development and Clinical Trials', 'Malaria - Elimination', 'Malaria - Epidemiology', 'Malaria - Genetics, Genomics and Evolution', 'Malaria - Immunology', 'Malaria – Surveillance and Data Utilization', 'Malaria - Pathogenesis', 'Malaria - Prevention', 'Malaria - Parasite Transmission Biology', 'Malaria - Vaccines and Immunotherapeutics', 'Cestodes (including taeniasis and cysticercosis, echinococcosis/hydatid disease, and others)', 'Helminths – Nematodes – Intestinal Nematodes', 'Helminths – Nematodes – Filariasis (Epidemiology and Modeling)', 'Helminths – Nematodes – Filariasis (Other)', 'Helminths – Nematodes – Filariasis (Diagnostics and Therapeutics)', 'Helminths – Nematodes – Filariasis (Molecular Biology and Immunology)', 'Helminths – Nematodes – Filariasis (Treatment and Morbidity Management)', 'Helminths – Nematodes – Filariasis (Behavioral and Social Sciences)', 'Schistosomiasis and Other Trematodes – Diagnostics and Treatment', 'Schistosomiasis and Other Trematodes – Epidemiology and Control', 'Schistosomiasis and Other Trematodes – Immunology, Pathology, Cellular and Molecular Biology', 'Respiratory Infections', 'Viruses - Emerging Viral Diseases', 'Viruses - Epidemiology', 'Viruses - Evolution and Genomic Epidemiology', 'Viruses - Field and ecological studies of viruses, including surveillance and spillover risk and emergence', 'Viruses - Immunology', 'Viruses - Pathogenesis and Animal Models', 'Viruses - Therapeutics and Antiviral Drugs', 'Viruses - Transmission Biology', 'Viruses - Vaccine Clinical Trials', 'Water, Sanitation, Hygiene and Environmental Health']


































