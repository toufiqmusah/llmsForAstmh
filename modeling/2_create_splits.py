"""
2_create_splits.py

Create stratified 5-fold splits from the prepared data.

Usage:
    python 2_create_splits.py [--num_folds N]
    
    --num_folds: number of folds (default: 5)
"""

import argparse
import pandas as pd
from pathlib import Path
from configs.config import (
    EMBEDDINGS_PARQUET,
    SPLITS_DIR,
    NUM_FOLDS,
    SEED,
)
from configs.data_utils import create_stratified_splits


def main():
    parser = argparse.ArgumentParser(description="Create stratified k-fold splits")
    parser.add_argument(
        "--num_folds",
        type=int,
        default=NUM_FOLDS,
        help="Number of folds",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 2: CREATE STRATIFIED SPLITS")
    print("=" * 80)

    # Load prepared data
    print(f"Loading embeddings from {EMBEDDINGS_PARQUET}...")
    df = pd.read_parquet(EMBEDDINGS_PARQUET)
    print(f"Loaded data shape: {df.shape}")

    # Create splits
    splits = create_stratified_splits(
        df=df,
        output_dir=SPLITS_DIR,
        num_folds=args.num_folds,
        seed=SEED,
        label_col="label_idx",
    )

    print("\n✓ Splits created and saved")
    print(f"  Output directory: {SPLITS_DIR}")
    print(f"  Number of folds: {args.num_folds}")
    print(f"  Splits metadata: {SPLITS_DIR / 'splits.pkl'}")


if __name__ == "__main__":
    main()
