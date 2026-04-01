# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:23:49 2024

@author: CharlesDelahunt

Goal: populate a keyword feature vector
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

def populateKeywordFeatureVector_fn(abstract, keywords, weights=[0, 1]):
    """
    Generate a keyword feature, as a sum of two ways:
        1. Give a 1 or 0 per keyword, according to whether it appears in the abstract
        2. Give an integer of the number of times it appears
    Use case insensitive matches (important special case: 'An.' for Anopheles, should
    still work as lower because of the '.')

    Parameters
    ----------
    abstract : str
    keywordList : list of str
    weights : list-like (length 2) of ints or floats. 1st value weights the binary feature, the
              2nd value weights the total count feature (2nd value is likely 0 or 1).

    Returns
    -------
    keywordFeature : np.array of floats

    """
    caseSensitiveKeywords = ['NTD', 'TB', 'HAT', 'STI', 'WASH', 'IBD', 'IBS', 'WHO',
                             'TLE', 'AIDS', 'HIV', 'SARS', 'CHAMPS', 'LAMP', 'An.']
    abstractLower = abstract.lower()
    binaryF = np.zeros(len(keywords))
    countF = np.zeros(len(keywords))

    # Binary feature:
    if weights[0]:
        for k in range(len(keywords)):
            this = keywords[k]
            if this in caseSensitiveKeywords:
                binaryF[k] = int(this in abstract)
            else:
                binaryF[k] = int(this.lower() in abstractLower)
    else:
        pass

    # Total count feature:
    if weights[1]:
        for k in range(len(keywords)):
            this = keywords[k]
            if this in caseSensitiveKeywords:
                ab = abstract
            else:
                this = this.lower()
                ab = abstractLower
            count = 0
            ab = abstract
            while len(ab) > 0:
                ind = ab.find(this)
                if ind >= 0:  # ie non-empty
                    count += 1
                    ab = ab[ind + 1:]
                else:
                    ab = ''
            countF[k] = count
    else:
        pass

    keywordFeature = weights[0] * binaryF + weights[1] * countF

    return keywordFeature

# End of populateKeywordFeatureVector_fn
#-------------------------------------------------------------------------

def create53ClassConfusionMatrix_fn(trueLabel, pred, testTrueLabel=[], testPred=[], titleStr=''):

    """
    Applies to post 17 April, eg results from 'astdm2023AbstractContents_18april2024.xlsx'

    Plot a confusion matrix for 51 fine-grained categories. Plot jittered points rather than numbers,
    with the important categories outlined in red.
    If 'testPred' is NOT empty, train and test set results are plotted in different colors.
    This function does not save the figure to PNG (no particular reason).
    CAUTION: the 'trueLabels' and 'pred' are alphabetical over the 53 merged categories as
    follows. The plot formatting depends on this.

    0 : Arthropods/Entomology - Other
    1 : Bacteriology - Enteric Infections
    2 : Bacteriology - Other Bacterial Infections
    3 : Bacteriology - Systemic Infections
    4 : Cestodes
    5 : Clinical Tropical Medicine
    6 : Ectoparasite-Borne Disease - All
    7 : Global Health - Diversity Inclusion Decolonization and Human Rights
    8 : Global Health - Information/Communication/Technologies
    9 : Global Health - Other
    10 : Global Health - Security/Preparedness
    11 : HIV and Tropical Co-Infections
    12 : Helminths - Nematodes - Filariasis (Epidemiology and Modeling)
    13 : Helminths - Nematodes - Filariasis (Molecular Biology and Immunology)
    14 : Helminths - Nematodes - Filariasis (Other)
    15 : Helminths - Nematodes - Intestinal Nematodes
    16 : Integrated Control Measures for Neglected Tropical Diseases (NTDs)
    17 : Kinetoplastida - All
    18 : Malaria - Antimalarial Resistance and Chemotherapy
    19 : Malaria - Diagnosis - Challenges and Innovations
    20 : Malaria - Drug Development and Clinical Trials
    21 : Malaria - Elimination
    22 : Malaria - Epidemiology
    23 : Malaria - Genetics Genomics and Evolution
    24 : Malaria - Immunology
    25 : Malaria - Parasite Transmission Biology
    26 : Malaria - Pathogenesis
    27 : Malaria - Prevention
    28 : Malaria - Surveillance and Data Utilization
    29 : Malaria - Unknown
    30 : Malaria - Vaccines and Immunotherapeutics
    31 : Mosquitoes - Biology Physiology and Immunity
    32 : Mosquitoes - Biology and Genetics of Insecticide Resistance
    33 : Mosquitoes - Bionomics Behavior and Surveillance
    34 : Mosquitoes - Epidemiology and Vector Control
    35 : Mosquitoes - Molecular Biology Population Genetics and Genomics
    36 : Mosquitoes - Unknown
    37 : One Health: Interconnections
    38 : Pneumonia Respiratory Infections and Tuberculosis
    39 : Schistosomiasis and Other Trematodes - Diagnostics and Treatment
    40 : Schistosomiasis and Other Trematodes - Epidemiology and Control
    41 : Schistosomiasis and Other Trematodes - Immunology Etc
    42 : Viruses - Emerging Viral Diseases
    43 : Viruses - Epidemiology
    44 : Viruses - Evolution and Genomic Epidemiology
    45 : Viruses - Field and ecological studies of viruses
    46 : Viruses - Immunology
    47 : Viruses - Pathogenesis and Animal Models
    48 : Viruses - Therapeutics and Antiviral Drugs
    49 : Viruses - Transmission Biology
    50 : Viruses - Unknown
    51 : Viruses - Vaccine Clinical Trials
    52 : Water Sanitation Hygiene and Environmental Health

    Parameters
    ----------
    trueLabels : list-like of ints
    pred : list-like of ints
    testTrueLabel : list-like of ints
    testPred : list-like of ints

    Returns
    -------
    None.

    """
    plt.figure()
    half = 0.4
    jitterX = -half + 2 *half *  np.random.random(len(pred))
    jitterY = -half + 2 * half * np.random.random(len(pred))
    jitterXTest = -half + 2 *half *  np.random.random(len(testPred))
    jitterYTest = -half + 2 * half * np.random.random(len(testPred))
    borders = np.array([0, 3,4,5, 6, 10, 11,15,16,17, 30, 36, 37, 38, 41,51]) + 0.5
    importantBorders = np.array([6, 10, 17, 30, 36, 37]) + 0.5
    plt.hlines(borders, -0.5, 50.5, colors='g', linestyles=':')
    plt.vlines(borders, -0.5, 50.5, color='g', linestyle=':')
    plt.hlines(importantBorders, -0.5, 50.5, color='r')
    for i in range(0, len(importantBorders), 2):
        plt.vlines([importantBorders[i], importantBorders[i + 1]],
                   importantBorders[i], importantBorders[i + 1], color='r')
    plt.gca().set_aspect('equal')

    plt.plot(pred + jitterY, trueLabel + jitterX, 'b+')
    if len(testPred) > 0:
        plt.plot(testPred + jitterYTest, testTrueLabel + jitterXTest, 'r+')

    xTickLocations = [0,2,4,5,6,8.5,11, 13.5,16,17,24,33.5,37,38,40,46.5, 52]
    xTickLabels = ['arthropods', 'bacteria', 'cestodes', 'clin trop med', 'ectoparasite',
                   'global health', 'helminths', 'HIV', 'integrated', 'kinetoplastida',
                   'malaria', 'mosquitoes', 'one health', 'pneumonia', 'schisto',
                   'viruses', 'sanitation']
    plt.yticks( xTickLocations, xTickLabels)
    plt.xticks( xTickLocations, xTickLabels, rotation='vertical')

    endStr = '.'
    if len(testPred) > 0:
        endStr = '. train = blue, test = red'
    plt.title('Predictions vs true class, ASTMH, ' + titleStr + endStr,
              fontweight='bold')
    plt.xlabel('true labels', fontweight='bold')
    plt.ylabel('predicted', fontweight='bold')
    plt.tight_layout()

# End of create53ClassConfusionMatrix_fn
#----------------------------------------------------------------------------

def map53ClassesTo17ShortGeneralCategories_fn(x):
    """
    Post 17 April.
    Convert 53-class integer labels to 17-category integer labels that are based on SHORT general
    categories, using the keys given in the confusion matrix functions above. This should also
    work for original General Categories, since the alphabetization is the same.

    Parameters
    ----------
    x : list-like of ints

    Returns
    -------
    x17

    """
    mapGranularToGeneral = \
    {0:0, 1:1, 2:1, 3:1, 4:2, 5:3, 6:4, 7:5, 8:5, 9:5, 10:5, 11:6, 12:7, 13:7, 14:7, 15:7, 16:8,
     17:9, 18:10, 19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10, 29:10,
     30:10, 31:11, 32:11, 33:11, 34:11, 35:11, 36:11, 37:12, 38:13, 39:14, 40:14, 41:14, 42:15,
     43:15, 44:15, 45:15, 46:15, 47:15, 48:15, 49:15, 50:15, 51:15, 52:16}

    a = np.array(x)
    x17 = a.copy()
    for k in mapGranularToGeneral.keys():
        x17[a == k] = mapGranularToGeneral[k]

    return x17

# End of map53ClassesTo17ShortGeneralCategories_fn
#----------------------------------------------------------------

def create51ClassConfusionMatrix_fn(trueLabel, pred, testTrueLabel=[], testPred=[], titleStr=''):
    """
    Plot a confusion matrix for 51 fine-grained categories. Plot jittered points rather than
    numbers, with the important categories outlined in red.
    If 'testPred' is NOT empty, train and test set results are plotted in different colors.
    This function does not save the figure to PNG (no particular reason).
    CAUTION: the 'trueLabels' and 'pred' are alphabetical over the 51 merged categories as
    follows. The plot formatting depends on this.

    0 : Arthropods/Entomology - Other
    1 : Bacteriology - Enteric Infections
    2 : Bacteriology - Other Bacterial Infections
    3 : Bacteriology - Systemic Infections
    4 : Cestodes
    5 : Clinical Tropical Medicine
    6 : Ectoparasite-Borne Disease - All
    7 : Global Health - Diversity, Inclusion, Decolonization and Human Rights
    8 : Global Health - Information/Communication/Technologies
    9 : Global Health - Other
    10 : Global Health - Planetary Health including Climate Change
    11 : Global Health - Security/Preparedness
    12 : HIV and Tropical Co-Infections
    13 : Helminths - Nematodes - Filariasis (Epidemiology and Modeling)
    14 : Helminths - Nematodes - Filariasis (Molecular Biology and Immunology)
    15 : Helminths - Nematodes - Filariasis (Other)
    16 : Helminths - Nematodes - Intestinal Nematodes
    17 : Integrated Control Measures for Neglected Tropical Diseases (NTDs)
    18 : Kinetoplastida - All
    19 : Malaria - Antimalarial Resistance and Chemotherapy
    20 : Malaria - Diagnosis - Challenges and Innovations
    21 : Malaria - Drug Development and Clinical Trials
    22 : Malaria - Elimination
    23 : Malaria - Epidemiology
    24 : Malaria - Genetics, Genomics and Evolution
    25 : Malaria - Immunology
    26 : Malaria - Parasite Transmission Biology
    27 : Malaria - Pathogenesis
    28 : Malaria - Prevention
    29 : Malaria - Surveillance and Data Utilization
    30 : Malaria - Vaccines and Immunotherapeutics
    31 : Mosquitoes - Biology and Genetics of Insecticide Resistance
    32 : Mosquitoes - Biology, Physiology and Immunity
    33 : Mosquitoes - Bionomics, Behavior and Surveillance
    34 : Mosquitoes - Epidemiology and Vector Control
    35 : Mosquitoes - Molecular Biology, Population Genetics and Genomics
    36 : One Health: Interconnections
    37 : Pneumonia, Respiratory Infections and Tuberculosis
    38 : Schistosomiasis and Other Trematodes - Diagnostics and Treatment
    39 : Schistosomiasis and Other Trematodes - Epidemiology and Control
    40 : Schistosomiasis and Other Trematodes - Immunology Etc
    41 : Viruses - Emerging Viral Diseases
    42 : Viruses - Epidemiology
    43 : Viruses - Evolution and Genomic Epidemiology
    44 : Viruses - Field and ecological studies of viruses
    45 : Viruses - Immunology
    46 : Viruses - Pathogenesis and Animal Models
    47 : Viruses - Therapeutics and Antiviral Drugs
    48 : Viruses - Transmission Biology
    49 : Viruses - Vaccine Clinical Trials
    50 : Water, Sanitation, Hygiene and Environmental Health

    Parameters
    ----------
    trueLabels : list-like of ints
    pred : list-like of ints
    testTrueLabel : list-like of ints
    testPred : list-like of ints

    Returns
    -------
    None.

    """
    plt.figure()
    half = 0.4
    jitterX = -half + 2 *half *  np.random.random(len(pred))
    jitterY = -half + 2 * half * np.random.random(len(pred))
    jitterXTest = -half + 2 *half *  np.random.random(len(testPred))
    jitterYTest = -half + 2 * half * np.random.random(len(testPred))
    borders = np.array([0, 3,4,5, 6, 11, 15,16,17,18, 30, 35, 36, 37, 40, 49]) + 0.5
    importantBorders = np.array([6, 11, 18, 30, 35, 36]) + 0.5
    plt.hlines(borders, -0.5, 50.5, colors='g', linestyles=':')
    plt.vlines(borders, -0.5, 50.5, color='g', linestyle=':')
    plt.hlines(importantBorders, -0.5, 50.5, color='r')
    for i in range(0, len(importantBorders), 2):
        plt.vlines([importantBorders[i], importantBorders[i + 1]],
                   importantBorders[i], importantBorders[i + 1], color='r')
    plt.gca().set_aspect('equal')

    plt.plot(pred + jitterY, trueLabel + jitterX, 'b+')
    if len(testPred) > 0:
        plt.plot(testPred + jitterYTest, testTrueLabel + jitterXTest, 'r+')

    xTickLocations = [0,2, 4,5,6,9,13.5, 16,17,18,25,33,36,37,39,44,50]
    xTickLabels = ['arthropods', 'bacteria', 'cestodes', 'clin trop med', 'ectoparasite',
                   'global health', 'helminths', 'HIV', 'integrated', 'kinetoplastida',
                   'malaria', 'mosquitoes', 'one health', 'pneumonia', 'schisto',
                   'viruses', 'sanitation']
    plt.yticks( xTickLocations, xTickLabels)
    plt.xticks( xTickLocations, xTickLabels, rotation='vertical')

    endStr = '.'
    if len(testPred) > 0:
        endStr = '. train = blue, test = red'
    plt.title('Predictions vs true class, ASTMH, ' + titleStr + endStr,
              fontweight='bold')
    plt.xlabel('true labels', fontweight='bold')
    plt.ylabel('predicted', fontweight='bold')
    plt.tight_layout()

# End of create51ClassConfusionMatrix_fn
#--------------------------------------------------------------

def create17ClassConfusionMatrixUsingSHORTGeneralCategoryLabels_fn(trueLabel, pred,
                                                                   testTrueLabel=[], testPred=[],
                                                                   titleStr=''):
    """
    Plot a confusion matrix for 17 general categories. Plot jittered points rather than numbers,
    with the important categories outlined in red.
    If 'testPred' is NOT empty, train and test set results are plotted in different colors.
    This function does not save the figure to PNG (no particular reason).
    CAUTION: the 'trueLabels' and 'pred' are alphabetical over the 17 short general categories as
    follows. The plot formatting depends on this.

    Short general categories:
    0 : Arthropods
    1 : Bacteriology
    2 : Cestodes
    3 : Clinical Trop Med
    4 : Ectoparasite-Borne
    5 : Global Health
    6 : HIV
    7 : Helminths
    8 : Integrated Control
    9 : Kinetoplastida
    10 : Malaria
    11 : Mosquitoes
    12 : One Health
    13 : Pneumonia TB
    14 : Schistosomiasis
    15 : Viruses
    16 : Water Sanitation

    Parameters
    ----------
    trueLabel : list-like of ints
    pred : list-like of ints
    testTrueLabel : list-like of ints
    testPred : list-like of ints

    Returns
    -------
    None.

    """
    plt.figure()
    half = 0.4
    jitterX = -half + 2 *half *  np.random.random(len(pred))
    jitterY = -half + 2 * half * np.random.random(len(pred))
    jitterXTest = -half + 2 *half *  np.random.random(len(testPred))
    jitterYTest = -half + 2 * half * np.random.random(len(testPred))
    borders = np.arange(0, 16) + 0.5
    importantBorders = np.array([4, 5, 7, 8, 9, 10, 11, 12]) + 0.5  # malaria, global health, integrated
    # control, one health
    plt.hlines(borders, -0.5, 16.5, color='g', linestyle=':')
    plt.vlines(borders, -0.5, 16.5, color='g', linestyle=':')
    plt.vlines(importantBorders, -0.5, 16.5, color='r')
    for i in range(0, len(importantBorders), 2):
        plt.vlines([importantBorders[i], importantBorders[i + 1]],
                   importantBorders[i], importantBorders[i + 1], color='r')
    plt.gca().set_aspect('equal')
    plt.plot(pred + jitterY, trueLabel + jitterX, 'b+')
    if len(testPred) > 0:
        plt.plot(testPred + jitterYTest, testTrueLabel + jitterXTest, 'r+')

    xTickLocations = np.arange(0, 17)
    xTickLabels = ['Arthropods', 'Bacteriology', 'Cestodes', 'Clinical Trop Med',
                   'Ectoparasite-Borne', 'Global Health', 'HIV', 'Helminths',
                   'Integrated Control', 'Kinetoplastida', 'Malaria', 'Mosquitoes',
                   'One Health', 'Pneumonia TB', 'Schistosomiasis', 'Viruses',
                   'Water Sanitation']
    plt.yticks( xTickLocations, xTickLabels)
    plt.xticks( xTickLocations, xTickLabels, rotation='vertical')

    endStr = '.'
    if len(testPred) > 0:
        endStr = '. train = blue, test = red'
    plt.title('Predictions vs true class, ASTMH, ' + titleStr + endStr,
              fontweight='bold')
    plt.xlabel('true labels', fontweight='bold')
    plt.ylabel('predicted', fontweight='bold')
    plt.tight_layout()

# End of create17ClassConfusionMatrixUsingSHORTGeneralCategoryLabels_fn
#----------------------------------------------------------------

def create17ClassConfusionMatrixWithCountsUsingSHORTGeneralCategoryLabels_fn(trueLabel,
                                                                             pred, titleStr=''):
    """
    Plot a confusion matrix for 17 general categories. In each box put numbers,
    with the important categories outlined in red. There is only one set of predictions-ground
    truth (no distinction between train and test set results).
    This function does not save the figure to PNG (no particular reason).
    CAUTION: the 'trueLabels' and 'pred' are alphabetical over the 17 short general categories as
    follows. The plot formatting depends on this.

    Short general categories (Post 17 april):
    0 : Arthropods
    1 : Bacteriology
    2 : Cestodes
    3 : Clinical Trop Med
    4 : Ectoparasite-Borne
    5 : Global Health
    6 : HIV
    7 : Helminths
    8 : Integrated Control
    9 : Kinetoplastida
    10 : Malaria
    11 : Mosquitoes
    12 : One Health
    13 : Pneumonia TB
    14 : Schistosomiasis
    15 : Viruses
    16 : Water Sanitation

    Parameters
    ----------
    trueLabel : list-like of ints
    pred : list-like of ints
    testTrueLabel : list-like of ints
    testPred : list-like of ints

    Returns
    -------
    None.

    """
    plt.figure()

    borders = np.arange(0, 16) + 0.5
    importantBorders = np.array([4, 5, 7, 8, 9, 10, 11, 12]) + 0.5  # malaria, global health, integrated
    # control
    plt.hlines(borders, -0.5, 16.5, color='g', linestyle=':')
    plt.vlines(borders, -0.5, 16.5, color='g', linestyle=':')
    plt.vlines(importantBorders, -0.5, 16.5, color='r')
    for i in range(0, len(importantBorders), 2):
        plt.vlines([importantBorders[i], importantBorders[i + 1]],
                   importantBorders[i], importantBorders[i + 1], color='r')
    plt.gca().set_aspect('equal')

    for i in range(17):
        for j in range(17):
            n = np.sum(np.logical_and(trueLabel == j, pred == i))
            xOffset = 0.1
            if n > 9:
                xOffset = 0.3
            if n > 99:
                xOffset = 0.45
            plt.text(i - xOffset, j - 0.25, str(n), fontweight='bold')

    xTickLocations = np.arange(0, 17)
    xTickLabels = ['Arthropods', 'Bacteriology', 'Cestodes', 'Clinical Trop Med',
                   'Ectoparasite-Borne', 'Global Health', 'HIV', 'Helminths',
                   'Integrated Control', 'Kinetoplastida', 'Malaria', 'Mosquitoes',
                   'One Health', 'Pneumonia TB', 'Schistosomiasis', 'Viruses',
                   'Water Sanitation']
    plt.yticks(xTickLocations, xTickLabels)
    plt.xticks(xTickLocations, xTickLabels, rotation='vertical')

    plt.title('Predictions vs true class, ASTMH, ' + titleStr + '.', fontweight='bold')
    plt.xlabel('true labels', fontweight='bold')
    plt.ylabel('predicted', fontweight='bold')
    plt.tight_layout()

# End of create17ClassConfusionMatrixWithCountsUsingSHORTGeneralCategoryLabels_fn
#----------------------------------------------------------------

def map51ClassesTo17ShortGeneralCategories_fn(x):
    """
    Convert 51-class integer labels to 17-category integer labels that are based on SHORT general
    categories, using the keys given in the confusion matrix functions above. This should also
    work for original General Categories, since the alphabetization is the same.

    Parameters
    ----------
    x : list-like of ints

    Returns
    -------
    x17

    """
    mapGranularToGeneral = \
    {0:0, 1:1, 2:1, 3:1, 4:2, 5:3, 6:4, 7:5, 8:5, 9:5, 10:5, 11:5, 12:6, 13:7, 14:7, 15:7, 16:7,
     17:8, 18:9, 19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10, 27:10, 28:10, 29:10,
     30:10, 31:11, 32:11, 33:11, 34:11, 35:11, 36:12, 37:13, 38:14, 39:14, 40:14, 41:15, 42:15,
     43:15, 44:15, 45:15, 46:15, 47:15, 48:15, 49:15, 50:16}

    a = np.array(x)
    x17 = a.copy()
    for k in mapGranularToGeneral.keys():
        x17[a == k] = mapGranularToGeneral[k]

    return x17

# End of map51ClassesTo17ShortGeneralCategories_fn
#----------------------------------------------------------------


def mapPriorShortGenCatIntegersToPost17AprilShortGenCats_fn(x):
    """
    Convert 17-class integer labels based on alphabetical order of General Categories to 17-category
    integer labels that are based on alphabetical SHORT general categories. The two lists are given
    below. Different labels are marked with (***)

    Original General categories:
    0 : Arthropods Entomology    (***)
    1 : Bacteriology             (***)
    2 : Cestodes                 (***)
    3 : Clinical Trop Med        (***)
    4 : Ectoparasite-Borne       (***)
    5 : Global Health
    6 : Helminths                (***)
    7 : HIV                      (***)
    8 - 16 : same as for Short gen cats (see below)

    Old short general categories:
    0 : Bacteriology
    1 : Cestodes
    2 : Clinical Trop Med
    3 : Ectoparasite-Borne
    4 : Entomology
    5 : Global Health
    6 : HIV
    7 : Helminths
    8 : Integrated Control
    9 : Kinetoplastida
    10 : Malaria
    11 : Mosquitoes
    12 : One Health
    13 : Pneumonia TB
    14 : Schistosomiasis
    15 : Viruses
    16 : Water Sanitation

    New (post 17 april) short general categories:
    0 : Arthropods
    1 : Bacteriology
    2 : Cestodes
    3 : Clinical Trop Med
    4 : Ectoparasite-Borne
    5 : Global Health
    6 : HIV
    7 : Helminths
    8 : Integrated Control
    9 : Kinetoplastida
    10 : Malaria
    11 : Mosquitoes
    12 : One Health
    13 : Pneumonia TB
    14 : Schistosomiasis
    15 : Viruses
    16 : Water Sanitation

    Parameters
    ----------
    x : list-like of ints

    Returns
    -------
    x17

    """
    mapGranularToGeneral = \
    {4:0, 0:1, 1:2, 2:3, 3:4, 6:7, 7:6}

    a = np.array(x)
    xShort = a.copy()
    for k in mapGranularToGeneral.keys():
        xShort[a == k] = mapGranularToGeneral[k]

    return xShort

# End of mapOriginalGeneralCategoryIntegersToShortGeneralCategories_fn
#----------------------------------------------------------------

def mapOriginalGeneralCategoryIntegersToShortGeneralCategories_fn(x):
    """
    NOTE: THIS FUNCTION SHOULD BE OBSOLETE.
    Convert 17-class integer labels based on alphabetical order of General Categories to
    17-category integer labels that are based on alphabetical SHORT general categories. The two
    lists are given below. Different labels are marked with (***)

    Original General categories:
    0 : Arthropods Entomology    (***)
    1 : Bacteriology             (***)
    2 : Cestodes                 (***)
    3 : Clinical Trop Med        (***)
    4 : Ectoparasite-Borne       (***)
    5 : Global Health
    6 : Helminths                (***)
    7 : HIV                      (***)
    8 - 16 : same as for Short gen cats (see below)

    Short general categories:
    0 : Bacteriology
    1 : Cestodes
    2 : Clinical Trop Med
    3 : Ectoparasite-Borne
    4 : Entomology
    5 : Global Health
    6 : HIV
    7 : Helminths
    8 : Integrated Control
    9 : Kinetoplastida
    10 : Malaria
    11 : Mosquitoes
    12 : One Health
    13 : Pneumonia TB
    14 : Schistosomiasis
    15 : Viruses
    16 : Water Sanitation

    Parameters
    ----------
    x : list-like of ints

    Returns
    -------
    x17

    """
    mapGranularToGeneral = \
    {0:4, 1:0, 2:1, 3:2, 4:3, 6:7, 7:6}

    a = np.array(x)
    xShort = a.copy()
    for k in mapGranularToGeneral.keys():
        xShort[a == k] = mapGranularToGeneral[k]

    return xShort

# End of mapOriginalGeneralCategoryIntegersToShortGeneralCategories_fn
#----------------------------------------------------------------
