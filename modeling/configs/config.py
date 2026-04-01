"""
config.py - Configuration for ASTMH classification pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Source data
SOURCE_DATA_FILE = Path("/Users/toufiq/Desktop/llmsForAstmh/data/materialsFor2026Version/combinedAbstractContents_2023_2024_2025_19mar2026.xlsx")

# Processed data
EMBEDDINGS_PARQUET = DATA_DIR / "embeddings_with_labels.parquet"
SPLITS_DIR = DATA_DIR / "splits"

# Embedding config
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# Training config
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT = 0.4
DEFAULT_LAYER_DIMS = [512, 256]
DEFAULT_NUM_LAYERS = len(DEFAULT_LAYER_DIMS)

# Important classes (priority categories for loss function)
IMPORTANT_CATEGORIES = [
    'Clinical Tropical Medicine', 'Global Health - Other',
    'Malaria - Antimalarial Resistance',
    'Malaria - Diagnosis', 'Malaria - Drug Dev', 'Malaria - Elimination',
    'Malaria - Epidemiology', 'Malaria - Genetics', 'Malaria - Immunology',
    'Malaria - Parasite Biology', 'Malaria - Pathogenesis', 'Malaria - Prevention',
    'Malaria - Vaccines', 'Malaria – Surveillance',
    'NTDs Control', 'Elimination', 'One Health',
    'Viruses - Emerging', 'Viruses - Epidemiology',
    'Viruses - Evolution', 'Genomic Epidemiology', 'Viruses - Field studies',
    'Viruses - Immunology', 'Viruses - Pathogenesis', 'Animal Models',
    'Viruses - Therapeutics', 'Viruses - Transmission', 'Viruses - Vaccine Trials'
]

# Training
NUM_FOLDS = 5
SEED = 42
DEVICE = "cuda"

# Ensure directories exist
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
