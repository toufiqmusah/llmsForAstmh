"""
4_predict.py

Make predictions on test/validation sets using trained models.

Usage:
    python 4_predict.py --fold 0 [--output_dir RESULTS]
    
    --fold: fold number (0-4)
    --output_dir: directory to save results (default: results/)
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from pytorch_lightning import Trainer

from configs.config import (
    EMBEDDINGS_PARQUET,
    SPLITS_DIR,
    MODELS_DIR,
    DEVICE,
)
from configs.data_utils import get_data_loaders, get_embedding_columns
from models.classifier import ASTMHClassifier


def main():
    parser = argparse.ArgumentParser(description="Make predictions on test sets")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (0-4)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Output directory for predictions",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 4: MAKE PREDICTIONS")
    print("=" * 80)

    output_dir = args.output_dir / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading embeddings from {EMBEDDINGS_PARQUET}...")
    df = pd.read_parquet(EMBEDDINGS_PARQUET)
    embedding_cols = get_embedding_columns(df)

    # Load splits
    splits_file = SPLITS_DIR / "splits.pkl"
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)

    train_idx, val_idx = splits[args.fold]

    # Create val dataloader
    _, val_loader = get_data_loaders(
        df=df,
        train_indices=train_idx,
        val_indices=val_idx,
        embedding_cols=embedding_cols,
        batch_size=1024,
    )

    # Load model
    model_path = MODELS_DIR / f"fold_{args.fold}" / "model.pt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using 3_train.py")
        return

    num_classes = len(df["label_idx"].unique())
    model = ASTMHClassifier(
        input_dim=len(embedding_cols),
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"Loaded model from {model_path}")

    # Create trainer for inference
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
    )

    # Make predictions
    print("Making predictions...")
    predictions = trainer.predict(model, val_loader)

    # Aggregate results
    all_preds = []
    all_probs = []
    all_targets = []

    for batch_results in predictions:
        all_preds.append(batch_results["predictions"].cpu().numpy())
        all_probs.append(batch_results["probabilities"].cpu().numpy())
        all_targets.append(batch_results["targets"].cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    # Create results dataframe
    val_df = df.iloc[val_idx].reset_index(drop=True)
    results_df = val_df[[
        "title", "abstractText", "shortMergedCat", "label_idx"
    ]].copy()

    # Add predictions
    idx_to_label = {idx: label for label, idx in 
                   pd.Series(df["label_idx"].values, 
                           index=df["shortMergedCat"].values).to_dict().items()}
    
    results_df["pred_label_idx"] = preds
    results_df["pred_label"] = results_df["pred_label_idx"].map(idx_to_label)
    results_df["pred_confidence"] = probs[np.arange(len(preds)), preds]
    results_df["correct"] = (results_df["label_idx"] == results_df["pred_label_idx"]).astype(int)

    # Add top-2 predictions
    top2_indices = np.argsort(-probs, axis=1)[:, :2]
    results_df["top2_pred_labels"] = [
        [idx_to_label[i] for i in top2_indices[j]]
        for j in range(len(top2_indices))
    ]
    results_df["top2_pred_confidence"] = [
        [probs[j, i] for i in top2_indices[j]]
        for j in range(len(top2_indices))
    ]

    # Save results
    results_path = output_dir / "predictions.parquet"
    results_df.to_parquet(results_path)
    print(f"✓ Predictions saved to {results_path}")

    # Calculate metrics
    accuracy = np.mean(results_df["correct"])
    print(f"\nValidation Accuracy (Fold {args.fold}): {accuracy:.4f}")

    # Save summary
    summary = {
        "fold": args.fold,
        "num_samples": len(results_df),
        "accuracy": accuracy,
        "num_classes": num_classes,
    }

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Fold {args.fold} Results\n")
        f.write(f"Samples: {summary['num_samples']}\n")
        f.write(f"Accuracy: {summary['accuracy']:.4f}\n")

    print(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
