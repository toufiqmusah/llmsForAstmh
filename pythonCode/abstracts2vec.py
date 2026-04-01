"""
abstract2vec.py

A script that uses open-source Hugging Face sentence transformers to encode 2023
ASTMH abstracts for use in downstream categorization tasks. 

Author: Olivia Zahn 
"""
import os
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

#%%
dataFolder = dataFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\data"

# Configure the ArgumentParser
cli_description = 'Utilizes open source LLM to create embeddings of ASTMH abstracts'
parser = argparse.ArgumentParser(description=cli_description)
parser.add_argument('--modelname', type=str)
parser.add_argument('--data_split', type=str)

# Extract command line arguments
args = parser.parse_args()

BASE_DATA_DIR = dataFolder

train_file_name = f'train_split_{args.data_split}.xlsx'
test_file_name = f'test_split_{args.data_split}.xlsx'

TRAIN_DATA_PATH = f'{dataFolder}/{train_file_name}'
TEST_DATA_PATH = f'{dataFolder}/{test_file_name}'
HF_PATH = f'sentence-transformers/{args.modelname}'

# Make the model save directory
if not os.path.exists(f'{BASE_DATA_DIR}/saved_embeddings/{args.modelname}/'):
    os.mkdir(f'{BASE_DATA_DIR}/saved_embeddings/{args.modelname}/')

# Make the split save directory 
if not os.path.exists(f'{BASE_DATA_DIR}/saved_embeddings/{args.modelname}/{args.data_split}/'):
    os.mkdir(f'{BASE_DATA_DIR}/saved_embeddings/{args.modelname}/{args.data_split}/')
    
SPLIT_SAVE_PATH = f'{BASE_DATA_DIR}/saved_embeddings/{args.modelname}/{args.data_split}/'

model = SentenceTransformer(HF_PATH)

# Load the abstracts
train_df = pd.read_excel(TRAIN_DATA_PATH)
test_df = pd.read_excel(TEST_DATA_PATH)

# The abstracts that we'd like to encode
train_abstracts = train_df['abstractText']
test_abstracts = test_df['abstractText']

# Abstracts are encoded by calling model.encode()
train_abstract_embeddings = model.encode(train_abstracts)
test_abstract_embeddings = model.encode(test_abstracts)

np.save(f'{SPLIT_SAVE_PATH}/train_embeddings_{args.data_split}.npy', train_abstract_embeddings)
np.save(f'{SPLIT_SAVE_PATH}/test_embeddings_{args.data_split}.npy', test_abstract_embeddings)
