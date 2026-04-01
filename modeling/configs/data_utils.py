"""
data_utils.py - Data utilities for ASTMH classification pipeline

Handles loading data, generating embeddings, creating splits, and dataset classes.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pickle


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for embeddings and labels.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, label_to_idx: dict = None):
        """
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) - label indices or strings
            label_to_idx: dict mapping label names to indices
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels) if isinstance(labels[0], (int, np.integer)) else torch.LongTensor([label_to_idx[l] for l in labels])
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_and_embed_data(
    excel_file: Path,
    output_file: Path,
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    force_regenerate: bool = False,
) -> tuple:
    """
    Load data from Excel file, generate embeddings, and save to parquet with labels.

    Args:
        excel_file: path to source Excel file
        output_file: path to save parquet file with embeddings
        embedding_model: sentence-transformer model name
        batch_size: batch size for embedding generation
        force_regenerate: if True, regenerate embeddings even if output exists

    Returns:
        tuple: (df with embeddings, label_to_idx mapping, idx_to_label mapping)
    """
    # Load data
    print(f"Loading data from {excel_file}...")
    df = pd.read_excel(excel_file)

    # Create label mappings
    unique_labels = sorted(df["shortMergedCat"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    df["label_idx"] = df["shortMergedCat"].map(label_to_idx)

    # Check if embeddings already exist
    if output_file.exists() and not force_regenerate:
        print(f"Loading embeddings from {output_file}...")
        df = pd.read_parquet(output_file)
    else:
        # Generate embeddings
        print(f"Generating embeddings using {embedding_model}...")
        model = SentenceTransformer(embedding_model)
        
        texts = (df["title"].fillna("") + " " + df["abstractText"].fillna("")).values
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        # Add embeddings to dataframe
        embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        df[embedding_cols] = embeddings
        
        # Save to parquet
        print(f"Saving embeddings to {output_file}...")
        df.to_parquet(output_file)

    print(f"Data shape: {df.shape}")
    print(f"Unique labels: {len(label_to_idx)}")

    return df, label_to_idx, idx_to_label


def create_stratified_splits(
    df: pd.DataFrame,
    output_dir: Path,
    num_folds: int = 5,
    seed: int = 42,
    label_col: str = "label_idx",
) -> dict:
    """
    Create stratified k-fold splits and save to parquet files.

    Args:
        df: dataframe with data and labels
        output_dir: directory to save split files
        num_folds: number of folds
        seed: random seed
        label_col: name of label column

    Returns:
        dict mapping fold numbers to (train_indices, val_indices)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = {}

    print(f"Creating {num_folds}-fold stratified splits...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[label_col])):
        print(f"  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
        
        splits[fold] = (train_idx, val_idx)

        # Save individual folds for reference
        df.iloc[train_idx].to_parquet(output_dir / f"fold_{fold}_train.parquet")
        df.iloc[val_idx].to_parquet(output_dir / f"fold_{fold}_val.parquet")

    # Save splits mapping
    splits_file = output_dir / "splits.pkl"
    with open(splits_file, "wb") as f:
        pickle.dump(splits, f)
    
    print(f"Splits saved to {output_dir}")

    return splits


def get_data_loaders(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    embedding_cols: list,
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """
    Create DataLoaders for training and validation.

    Args:
        df: dataframe with embeddings and labels
        train_indices: indices for training data
        val_indices: indices for validation data
        embedding_cols: list of embedding column names
        batch_size: batch size
        num_workers: number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    train_embeddings = train_df[embedding_cols].values.astype(np.float32)
    train_labels = train_df["label_idx"].values.astype(np.int64)
    val_embeddings = val_df[embedding_cols].values.astype(np.float32)
    val_labels = val_df["label_idx"].values.astype(np.int64)

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_embedding_columns(df: pd.DataFrame) -> list:
    """
    Extract embedding column names from dataframe.
    """
    return [col for col in df.columns if col.startswith("emb_")]
