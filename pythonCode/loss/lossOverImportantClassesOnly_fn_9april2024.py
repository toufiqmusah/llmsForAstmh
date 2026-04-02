# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:37:18 2024

@author: CharlesDelahunt

Goal: nn.Module for custom loss function for ASTMH abstract sorting problem.

Given model logit predictions (output) and true classes (target) as inputs, the module:
    1. Defines 'keptCats', which are the categories of interest (Malaria, Global Health, and
       Integrated Control).
       ATTENTION!!! THESE VALUES ARE HARD-CODED HERE. SEE INLINE COMMENTS BELOW FOR DETAILS!
    2. Applies softmax to the output.
    3. Defines a "predicted" category for each sample according to the cat with the highest
       softmax score.
    4. Finds the samples to be ignored, ie those for which neither predicted cat nor target
       are in keptCats.
    5. Alters the softmax scores of these ignored samples to be 1 for the target class and 0
       elsewhere, ie make them perfect predictions so that they do not contribute anything to
       the loss function.
    6. Applies the built-in 'loss' function (CAUTION: I don't know if this bit is correct; for
       example, does the 'keepSoftmax' argin need to be recast to a tensor? Is 'loss' called
       correctly?).

"""
import torch
from torch import nn
import numpy as np

class GreenCatNLLLoss(nn.Module):
    """
    doc string
    """
    def __init__(self):
        super(GreenCatNLLLoss, self).__init__()

        self.loss = nn.NLLLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%% Softmax function for an array:
    def softmax_fn(self, logits):
        """
        Apply softmax to an array of logits (model outputs). Array is n x c where c is number
        of classes, and each row is a sample's scores.

        Parameters
        ----------
        logits : np.array, n x c

        Returns
        -------
        softmaxValues : np.array, same shape as logits

        """

        t = torch.exp(logits)
        input_max, max_indices = torch.max(t, axis=1)
        maxVals = torch.unsqueeze(input_max, 1) #np.expand_dims(maxVals, 1)
        t = t - torch.tile(maxVals, (1, t.shape[1])) #np.tile(maxVals, (1, t.shape[1])) 
        # copilot says that subtracting the
        # max improves numerical stability
        rowSums = torch.sum(t, axis=1) #np.sum(t, axis=1)
        rowSums = torch.unsqueeze(rowSums, 1) #np.expand_dims(rowSums, 1)
        softmaxValues = t / torch.tile(rowSums, (1, t.shape[1])) #np.tile(rowSums, (1, t.shape[1]))  # normalize each row to 1

        return softmaxValues

    #%%
    def forward(self, output, target):
        """
        Calculates a loss function by concentrating only on those abstracts whose original or 
        predicted classes are in the priority categories (ie the ones we are expected to review).
        Read CRUCIAL notes below about sorted order of categories.
        
        2026: These are listed in 
        'data/materialsFor2026Version/ASTMH 2025 and 2026 Abstract Submission Categories.xlsx", and
        when converted to shortMergedCats are:
        shortMergedCatsToVet2026 = \
            ['Clinical Tropical Medicine', 'Global Health - Other', 
             'Malaria - Antimalarial Resistance', 
             'Malaria - Diagnosis', 'Malaria - Drug Dev, Trials', 'Malaria - Elimination', 
             'Malaria - Epidemiology', 'Malaria - Genetics', 'Malaria - Immunology', 
             'Malaria - Parasite Biology', 'Malaria - Pathogenesis', 'Malaria - Prevention',  
             'Malaria - Vaccines', 'Malaria – Surveillance', 'NTDs Control', 'Elimination', 
             'One Health', 'Viruses - Emerging', 'Viruses - Epidemiology', 
             'Viruses - Evolution', 'Genomic Epidemiology', 'Viruses - Field studies', 
             'Viruses - Immunology', 'Viruses - Pathogenesis', 'Animal Models', 
             'Viruses - Therapeutics', 'Viruses - Transmission', 'Viruses - Vaccine Trials']
        This is a subset of 'allShortMergedCatsAlphabetical' found by:
            df = pd.read_excel(r"data\materialsFor2026Version" + 
                           r"\combinedAbstractContents_2023_2024_2025_19mar2026.xlsx")
            allShortMergedCatsAlphabetical = np.unique(df['shortMergedCat'].values)
            
        Parameters:
            output: torch.Tensor
            target: torch.Tensor. 
            
            !!! CRUCIAL !!! For 2026, 'target' is a Tensor version of 
                'allShortMergedCatsAlphabetical' given above.
            !!! CRUCIAL !!! We assume the shortMergedCats are ALPHABETICAL. This happens if 
                they are collected using np.unique(), but torch.unique() might not sort them.
        

        Returns:
            loss: torch.Tensor - loss.
        """
        softmax = output
        # 1. Define the labels of classes to keep, eg Malaria, NTDs, and Global Health. This will 
        # vary by year.       
        
        # 2026: 
        
        # The priority list has the following indices in 'allShortMergedCatsAlphabetical':
        if output.shape[1] == 54:
            keptCats = [6,11,21,22,23,24,25,26,27,28,29,30,31,32,38,39,44,45,46,47,48,49,50,51,52]  
        else:  # case: intra-class, eg Malaria (12 classes), Global Health (5 classes)
            keptCats = torch.unique(target) #np.unique(target)  # ie keep all

        # 2. Calculate softmaxes:
        #softmax = self.softmax_fn(output) #np.array(output)
        # 3. Choose a predicted class for each output sample via its max logit score (since
        # this would give the highest softmax score also), and record if that prediction is
        # in keptClasses:
        t, max_indices = torch.max(softmax, axis=1) #np.max(softmax, axis=1)  # max of each row
        keepPred = np.ones(t.shape[0], dtype=bool)
        for i in range(len(t)):
            # print(torch.where(softmax[i,:] == t[i]))
            keepPred[i] = torch.where(softmax[i,:] == t[i])[0][0] in keptCats 
            # np.where(softmax[i,:] == t[i])[0][0] in keepClasses  # if two logit values are exactly
            # equal, we'll use the first one (hopefully this does not arise).
            # print(torch.where(softmax[i,:] == t[i]))

        # Record whether the target class is in keptClasses:
        keepTarget = np.ones(target.shape, dtype=bool)
        for i in range(len(t)):
            keepTarget[i] = target[i] in keptCats

        # 4. Find samples to ignore, ie where neither the predicted nor the target class is in
        # 'keepClasses'. These are the samples we want the loss function to ignore. 
        ignore = np.where(np.logical_not(np.logical_or(keepPred, keepTarget)))[0]
        ignore_mat = torch.ones(softmax.shape)        
        for i in ignore:
            ignore_mat[i,:] = 0
            
        # 5. Make the predictions for ignored samples perfect, ie 0.947 in the target class and 0.001
        # everywhere else (if 54 classes):        
        adjust_mat = torch.zeros(softmax.shape)
        for i in ignore:
            adjust_mat[i, :] = 0.001
            adjust_mat[i, target[i]] = 1 - ((softmax.shape[1]-1) * 0.001) 
            
        ignore_mat = ignore_mat.to(self.device)
        adjust_mat = adjust_mat.to(self.device)

        # Replace the ignored samples' softmax scores with adjust_mat:
        keepSoftmax = (softmax * ignore_mat) + adjust_mat  
        
        # 6. Apply the loss function:
        logKeepSoftmax = torch.log(keepSoftmax) # Because NLLLoss takes log-softmax as input. See
        # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/8
        loss = self.loss(logKeepSoftmax, target)

        return loss
     
    #%% Priority list from previous year:
    # 2025: These are hard-coded labels from 'astmhSupportFunctions_6april2024' 
    # if output.shape[1] == 51:  # case: 51 classes #OTZ EDIT: changed logits to output
    #     keepClasses = list(range(7, 12)) + [17] + list(range(19, 31))
    # elif output.shape[1] == 17:  # Case: 2025, just using general categories
    #     keepClasses = [5, 8, 10]
