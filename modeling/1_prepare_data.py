"""
1_prepare_data.py

Load data from Excel, generate embeddings, and save to parquet with labels.

Usage:
    python 1_prepare_data.py [--force]
    
    --force: regenerate embeddings even if output exists
"""

import argparse
from pathlib import Path
from configs.config import (
    SOURCE_DATA_FILE,
    EMBEDDINGS_PARQUET,
    EMBEDDING_MODEL,
)
from configs.data_utils import load_and_embed_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data: load Excel, generate embeddings, save to parquet"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of embeddings",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1: PREPARE DATA & GENERATE EMBEDDINGS")
    print("=" * 80)

    df, label_to_idx, idx_to_label = load_and_embed_data(
        excel_file=SOURCE_DATA_FILE,
        output_file=EMBEDDINGS_PARQUET,
        embedding_model=EMBEDDING_MODEL,
        force_regenerate=args.force,
    )

    print("\n✓ Data prepared and embeddings saved to parquet")
    print(f"  File: {EMBEDDINGS_PARQUET}")
    print(f"  Shape: {df.shape}")
    print(f"  Classes: {len(label_to_idx)}")


if __name__ == "__main__":
    main()
