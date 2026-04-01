"""
EmbeddingData.py

PyTorch Dataset for use with EmbeddingData.py (LightningDataModule).

Author: Olivia Zahn 
"""
import os

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

# from sentence_transformers import SentenceTransformer  # commented out for 2025

class EmbeddingDataset(Dataset):
    """
    doc string
    """
    def __init__(self,
                 data_df: pd.DataFrame,
                 embed_path: os.PathLike,
                 hf_llm_name: str,
                 trainValTag: str = 'test', # 'train',
                 augmentations = ['', ''],
                 num_classes: int = 46, # 17,
                 w0: int = 0,
                 w1: int = 1,
                 ):
        
        embed_path = r"V:\FAMLI\Results\Olivia\astmh\dataToRunPredictOneTime"
        # setup 
        self.EMBED_PATH = embed_path
        self.data_df = data_df
        self.num_classes = num_classes
        self.augmentations = augmentations
        
        # First see if we're using pre-built embeddings:
        if not os.path.exists(embed_path):
            os.mkdir(embed_path)
        
        if False: # trainValTag == 'train':
            aug_tags = ''
            for aug_type in self.augmentations:
                aug_tags = aug_tags + aug_type + '_'
            embedding_base_filename = 'train_' + aug_tags + '_embeddings.npy'
        else:
            embedding_base_filename = "test_embeddings.npy"  # f"{trainValTag}_embeddings.npy"
            # Question: how to pass in args to give more detail to the embeddings filename?

        if os.path.exists(os.path.join(embed_path, embedding_base_filename)):
            # If the embedded abstracts file exists, load the embeddings.
            self.abstract_embeddings = np.load(os.path.join(embed_path, embedding_base_filename))
        
        # Otherwise, load the abstract text and create embeddings:
        else:
            # If we are working with the training data, load the preprocessed abstracts. 
            if trainValTag == 'train':
                abstracts = self.data_df['preprocessed_abstractText']
    
            else:
                abstracts = self.data_df['abstractText']
            # If we are doing 51-class classification, use "mergedCategory"
            if self.num_classes == 51: # for 2025, 46. For 2024, 51            
                categories = self.data_df["mergedCategory"].unique()
                self.cat_df = pd.DataFrame(categories, columns=['categories'])
            elif self.num_classes == 17:
                categories = np.unique(self.data_df['shortGenCat'])#self.data_df["shortGenCat"].unique()
                self.cat_df = pd.DataFrame(categories, columns=['categories'])
    
           
                # If embeddings do not exist, embed the abstracts and save them.
                HF_PATH = f'sentence-transformers/{hf_llm_name}'
                model = SentenceTransformer(HF_PATH)
    
                # Encode the abstracts
                self.abstract_embeddings = model.encode(abstracts)
                # Save the encodings. 
                np.save(os.path.join(embed_path, embedding_base_filename), 
                        self.abstract_embeddings)
            
            # # Comment out for 2025, since we are not using keywords (they did not work in 2024): 
            # # Import the keywords file and read as list of strings. 
            # kws_file = open('/workspace/code/data/keywords_8april2024.txt', 'r')
            # kws = kws_file.read()
            # self.kws_list = kws.split(',')
            # self.kws_list[-1] = self.kws_list[-1][:-1]
    
            # # Create the encoded keyword vector and save as column in df.
            # # Use 'abstractText', not 'preprocessed_abstractText'
            # self.data_df['kw_vector'] = self.data_df['abstractText'].apply(
            #     lambda x: self.populateKeywordFeatureVector_fn(x,
            #                                                     self.kws_list,
            #                                                     weights=[w0,w1]))

#%% defs:
        
    def __len__(self):
        """ Returns length of dataset. """
        return len(self.data_df)

    def __getitem__(self, idx):
        emb_i = torch.Tensor(self.abstract_embeddings[idx])

        if self.num_classes == 46:   # 46 for 2025, 51 for 2024:
            cat_str = self.data_df.iloc[idx]['mergedCategory']
        elif self.num_classes == 17:
            cat_str = self.data_df.iloc[idx]["shortGenCat"]
        label_i = []  # self.cat_df.loc[self.cat_df['categories'] == cat_str].index.values[0]
        kw_i = []  # [] for 2025. For 2024: torch.Tensor(self.data_df.iloc[idx]['kw_vector'])
        emb_kw_i = []  # [] for 2025. For 2024: torch.cat((emb_i, kw_i), 0)

        return emb_i, kw_i, emb_kw_i, label_i
    
#%% Another def:
        
    def populateKeywordFeatureVector_fn(self, abstract, keywords, weights=[0, 1]):
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

        

    