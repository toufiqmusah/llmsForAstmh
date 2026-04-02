"""
3_train_xgboost.py

Train XGBoost classifier on prepared embeddings.

XGBoost is often better for:
- Avoiding overfitting without as much tuning
- Fast training on tabular data (embeddings)
- Better generalization on validation sets

Usage:
    python 3_train_xgboost.py --fold 0 [--depth 6] [--lr 0.1] 
                               [--subsample 0.8] [--colsample 0.8]
                               [--num_rounds 500] [--early_stopping 50]

Examples:
    # Basic training (recommended default)
    python 3_train_xgboost.py --fold 0
    
    # Shallow trees (reduce overfitting)
    python 3_train_xgboost.py --fold 0 --depth 4 --subsample 0.7 --colsample 0.7
    
    # Deeper trees (more capacity)
    python 3_train_xgboost.py --fold 0 --depth 8 --subsample 0.9
    
    # More aggressive regularization
    python 3_train_xgboost.py --fold 0 --depth 5 --subsample 0.6 --colsample 0.6 --l2 5.0
    
    # Fast training with fewer rounds
    python 3_train_xgboost.py --fold 0 --num_rounds 200 --early_stopping 30
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from configs.config import (
    EMBEDDINGS_PARQUET,
    SPLITS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    SEED,
)
from configs.data_utils import get_embedding_columns
from models.xgboost_classifier import XGBoostClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier on embeddings"
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold number (0-4)")
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Max tree depth (default: 6, try 4-8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1, range: 0.01-0.5)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Fraction of samples per tree (default: 0.8, range: 0.5-1.0)",
    )
    parser.add_argument(
        "--colsample",
        type=float,
        default=0.8,
        help="Fraction of features per tree (default: 0.8, range: 0.5-1.0)",
    )
    parser.add_argument(
        "--min_child_weight",
        type=int,
        default=1,
        help="Min weight in child nodes (default: 1, try 1-5)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0,
        help="L2 regularization (default: 1.0, higher = more regularization)",
    )
    parser.add_argument(
        "--l1",
        type=float,
        default=0.0,
        help="L1 regularization (default: 0.0)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=500,
        help="Max boosting rounds (default: 500)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 3: TRAIN XGBOOST CLASSIFIER")
    print("=" * 80)

    # Load data
    print(f"Loading embeddings from {EMBEDDINGS_PARQUET}...")
    df = pd.read_parquet(EMBEDDINGS_PARQUET)
    embedding_cols = get_embedding_columns(df)
    print(f"Loaded shape: {df.shape}, embedding columns: {len(embedding_cols)}")

    # Load splits
    splits_file = SPLITS_DIR / "splits.pkl"
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)

    train_idx, val_idx = splits[args.fold]
    print(f"Fold {args.fold}: train={len(train_idx)}, val={len(val_idx)}")

    # Prepare data
    X_train = df.iloc[train_idx][embedding_cols].values
    y_train = df.iloc[train_idx]["label_idx"].values
    
    X_val = df.iloc[val_idx][embedding_cols].values
    y_val = df.iloc[val_idx]["label_idx"].values
    
    num_classes = len(df["label_idx"].unique())
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  Number of classes: {num_classes}")

    # Create model
    model = XGBoostClassifier(
        num_classes=num_classes,
        max_depth=args.depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        lambda_l2=args.l2,
        alpha_l1=args.l1,
        random_state=SEED,
        device=args.device,
    )

    print(f"\nModel config:")
    print(f"  Max depth: {args.depth}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Subsample: {args.subsample}")
    print(f"  Colsample bytree: {args.colsample}")
    print(f"  Min child weight: {args.min_child_weight}")
    print(f"  L2 regularization: {args.l2}")
    print(f"  L1 regularization: {args.l1}")
    print(f"  Num rounds: {args.num_rounds}")
    print(f"  Early stopping patience: {args.early_stopping}")

    # Train
    print("\nTraining...")
    train_result = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        num_rounds=args.num_rounds,
        early_stopping_rounds=args.early_stopping,
    )

    # Get predictions for detailed metrics
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    val_proba = model.predict_proba(X_val)

    print(f"\nClassification Report (Validation Set):")
    print(classification_report(y_val, val_preds))

    # Save model
    model_dir = MODELS_DIR / f"fold_{args.fold}_xgb"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.json"
    model.save(str(model_path))

    # Save config
    config_path = model_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Model type: XGBoost\n")
        f.write(f"Max depth: {args.depth}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Subsample: {args.subsample}\n")
        f.write(f"Colsample bytree: {args.colsample}\n")
        f.write(f"Min child weight: {args.min_child_weight}\n")
        f.write(f"L2 regularization: {args.l2}\n")
        f.write(f"L1 regularization: {args.l1}\n")
        f.write(f"Number of rounds: {args.num_rounds}\n")
        f.write(f"Early stopping patience: {args.early_stopping}\n")
        f.write(f"Best iteration: {train_result['best_iteration']}\n")
        f.write(f"\nResults:\n")
        f.write(f"Train accuracy: {train_result['train_acc']:.4f}\n")
        f.write(f"Val accuracy: {train_result['val_acc']:.4f}\n")

    # Save predictions
    pred_path = model_dir / "predictions.npz"
    np.savez(
        pred_path,
        val_predictions=val_preds,
        val_probabilities=val_proba,
        val_targets=y_val,
        train_predictions=train_preds,
        train_targets=y_train,
    )
    print(f"\n✓ Predictions saved to {pred_path}")

    # Save feature importance
    feature_importance = model.get_feature_importance(top_n=20)
    if feature_importance:
        importance_path = model_dir / "feature_importance.txt"
        with open(importance_path, "w") as f:
            f.write("Feature importance (top 20):\n")
            for feat, score in feature_importance.items():
                f.write(f"  {feat}: {score}\n")
        print(f"✓ Feature importance saved to {importance_path}")

    print(f"\n✓ All outputs saved to {model_dir}")


if __name__ == "__main__":
    main()
