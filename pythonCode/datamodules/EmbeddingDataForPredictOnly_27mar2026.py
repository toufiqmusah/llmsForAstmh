# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:13:06 2026 

This is a 2026 version of 'EmbeddingDataForPredictOnly_23apr2025.py'
 
EmbeddingData.py

LightningDataModule for training a multi-class classification model for
ASTMH abstract embeddings. 

Author: Olivia Zahn, modified CharlesDelahunt

Goal: A version of class EmbeddingData that does not require any operations on text (eg for 
training) 
"""
import os

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

# from datamodules.data_utils import synonym_replacement, random_insertion, random_deletion, \
#     random_masking
# from nltk.tokenize import word_tokenize, sent_tokenize
from datasets.EmbeddingDataset import EmbeddingDataset

#%%

class EmbeddingData(LightningDataModule):
    """
    doc string
    """
    def __init__(self,
                 batch_size: int = 1024,
                 data_split: str = '1_41824',   # !!! ATTN !!! Maybe needs to be changed for 2026
                 LLM_name: str = 'all-mpnet-base-v2',
                 augmentations: list = ['', ''], 
                 num_aug_words: int = 1,
                 num_classes: int = 54,  # 54 in 2026
                 data_dir: os.PathLike = '/workspace/code/data/',  # not relevant
                 embed_dir: os.PathLike = r"V:\FAMLI\Results\Olivia\astmh",  # 2025 
                 #'/workspace/code/data/saved_embeddings/',
                 w0: int = 0,
                 w1: int = 1,
                 ):
        super().__init__()

        self.train_dl_options = {
            'batch_size': batch_size,
            'shuffle': True,
            }
        self.val_dl_options = {
            'batch_size': batch_size,
            'shuffle': False,
            }
        self.predict_dl_options = {
            'batch_size': 1,
            'shuffle': False,
            }

        self.split = data_split
        self.LLM = LLM_name

        self.augmentations = augmentations
        self.n = num_aug_words
        self.num_classes = num_classes
        self.w0 = w0
        self.w1 = w1

        # LLM_EMBED_PATH = os.path.join(embed_dir, LLM_name)
        # if not os.path.exists(LLM_EMBED_PATH):
        #     os.mkdir(LLM_EMBED_PATH)

        # self.train_embed_path = os.path.join(embed_dir, LLM_name, self.split)
        # self.test_embed_path = os.path.join(embed_dir, LLM_name, self.split)
        self.test_embed_path = embed_dir  # os.path.join(embed_dir, self.split)
        
        # self.train_data_path = os.path.join(data_dir, f"train_split_{self.split}.xlsx")
        # self.test_data_path = os.path.join(data_dir, f"test_split_{self.split}.xlsx")

        # self.train_data_df = pd.read_excel(self.train_data_path)
        # self.test_data_df = pd.read_excel(self.test_data_path)
        self.test_data_df = \
            pd.read_excel(os.path.join(embed_dir, 
                                       "embedded_newAbstractsToReclassify_merged_2026.xlsx"))

        # self.LLM = LLM_name

        # # Apply any augmentations.
        # self.prepare_data()

    # def prepare_data(self):
    #     """
    #     Augmentations are only applied to training data. 
    #     Agmented abstracts are saved in a new column of dataframe "preprocessed_abstractText". 
    #     """
    #     self.train_data_df['preprocessed_abstractText'] = self.train_data_df['abstractText']
        
    #     if self.augmentations != []:
    #         to_aug_df = self.train_data_df

    #     if 'synonym_replacement' in self.augmentations:
    #         df_for_current_aug = to_aug_df
    #         df_for_current_aug['preprocessed_abstractText'] = to_aug_df['preprocessed_abstractText'].apply(synonym_replacement)
    #         self.train_data_df = pd.concat([self.train_data_df, df_for_current_aug], axis=0, ignore_index=True)
    #         del df_for_current_aug

    #     if 'random_insertion' in self.augmentations:
    #         df_for_current_aug = to_aug_df
    #         df_for_current_aug['preprocessed_abstractText'] = to_aug_df['preprocessed_abstractText'].apply(random_insertion)
    #         self.train_data_df = pd.concat([self.train_data_df, df_for_current_aug], axis=0, ignore_index=True)
    #         del df_for_current_aug

    #     if 'random_deletion' in self.augmentations:
    #         df_for_current_aug = to_aug_df
    #         df_for_current_aug['preprocessed_abstractText'] = to_aug_df['preprocessed_abstractText'].apply(random_deletion)
    #         self.train_data_df = pd.concat([self.train_data_df, df_for_current_aug], axis=0, ignore_index=True)
    #         del df_for_current_aug 

    #     if 'random_masking' in self.augmentations:
    #         df_for_current_aug = to_aug_df
    #         df_for_current_aug['preprocessed_abstractText'] = to_aug_df['preprocessed_abstractText'].apply(random_masking)
    #         self.train_data_df = pd.concat([self.train_data_df, df_for_current_aug], axis=0, ignore_index=True)
    #         del df_for_current_aug 


    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = EmbeddingDataset(data_df=self.train_data_df, 
                                                  embed_path=self.train_embed_path, 
                                                  hf_llm_name=self.LLM, 
                                                  trainValTag='train',
                                                  augmentations=self.augmentations,
                                                  num_classes=self.num_classes,
                                                  w0=self.w0,
                                                  w1=self.w1,)
            self.val_dataset = EmbeddingDataset(data_df=self.test_data_df, 
                                                embed_path=self.test_embed_path, 
                                                hf_llm_name=self.LLM, 
                                                trainValTag='test',
                                                augmentations=[],
                                                num_classes=self.num_classes,
                                                w0=self.w0,
                                                w1=self.w1,)

        if stage == 'eval':
            self.val_dataset = EmbeddingDataset(data_df=self.test_data_df, 
                                                embed_path=self.test_embed_path, 
                                                hf_llm_name=self.LLM, 
                                                trainValTag='test',
                                                augmentations=[],
                                                num_classes=self.num_classes,
                                                w0=self.w0,
                                                w1=self.w1,)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dl_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dl_options)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, **self.predict_dl_options)
    
