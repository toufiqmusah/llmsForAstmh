"""
3_train.py

Train the classifier on prepared embeddings.

Usage:
    python 3_train.py --fold 0 [--epochs 100] [--batch_size 1024] [--lr 1e-4] 
                      [--layer_dims 512 256] [--dropout 0.4] [--use_important_loss]
                      [--use_batch_norm] [--use_layer_norm] [--use_residual]
                      [--use_attention] [--activation relu]
                      [--wandb_project PROJECT] [--wandb_name NAME]
    
Examples:
    # Basic training
    python 3_train.py --fold 0
    
    # With batch norm and residual connections
    python 3_train.py --fold 0 --epochs 100 --lr 1e-4
    
    # With advanced features (attention, layer norm, no residual)
    python 3_train.py --fold 0 --use_attention --use_layer_norm --no_residual
    
    # With custom loss and WandB logging
    python 3_train.py --fold 0 --use_important_loss --wandb_project MyProject
    
    # Custom architecture with GELU activation
    python 3_train.py --fold 0 --layer_dims 768 512 256 --activation gelu
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from configs.config import (
    EMBEDDINGS_PARQUET,
    SPLITS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    DEVICE,
    SEED,
    IMPORTANT_CATEGORIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LAYER_DIMS,
    DEFAULT_DROPOUT,
)
from configs.data_utils import get_data_loaders, get_embedding_columns
from models.classifier import ASTMHClassifier
from loss.custom_loss import ImportantCategoryLoss


def main():
    parser = argparse.ArgumentParser(description="Train classifier on embeddings")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (0-4)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--layer_dims",
        type=int,
        nargs="+",
        default=DEFAULT_LAYER_DIMS,
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--use_important_loss",
        action="store_true",
        help="Use custom loss focusing on important categories",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--devices", type=str, default="0", help="GPU devices to use")
    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        help="Disable batch normalization",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Use layer normalization instead of batch norm",
    )
    parser.add_argument(
        "--no_residual",
        action="store_true",
        help="Disable residual connections",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Use feature attention mechanism",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "elu"],
        help="Activation function",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 3: TRAIN CLASSIFIER")
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

    # Create dataloaders
    train_loader, val_loader = get_data_loaders(
        df=df,
        train_indices=train_idx,
        val_indices=val_idx,
        embedding_cols=embedding_cols,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create loss function
    if args.use_important_loss:
        # Map important category names to indices
        label_to_idx = pd.Series(
            df["label_idx"].values, index=df["shortMergedCat"].values
        ).to_dict()
        important_indices = [
            label_to_idx[cat] for cat in IMPORTANT_CATEGORIES if cat in label_to_idx
        ]
        print(f"Using custom loss for {len(important_indices)} important categories")
        loss_fn = ImportantCategoryLoss(important_indices, device=device)
    else:
        loss_fn = torch.nn.NLLLoss()

    # Create model
    num_classes = len(df["label_idx"].unique())
    model = ASTMHClassifier(
        input_dim=len(embedding_cols),
        layer_dims=args.layer_dims,
        num_classes=num_classes,
        learning_rate=args.lr,
        dropout=args.dropout,
        loss_fn=loss_fn,
        use_batch_norm=not args.no_batch_norm,
        use_layer_norm=args.use_layer_norm,
        use_residual=not args.no_residual,
        use_attention=args.use_attention,
        activation=args.activation,
    )

    print(f"\nModel config:")
    print(f"  Input dim: {len(embedding_cols)}")
    print(f"  Hidden layers: {args.layer_dims}")
    print(f"  Output classes: {num_classes}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch norm: {not args.no_batch_norm}")
    print(f"  Layer norm: {args.use_layer_norm}")
    print(f"  Residual: {not args.no_residual}")
    print(f"  Attention: {args.use_attention}")
    print(f"  Activation: {args.activation}")

    # Setup logging
    loggers = []
    if args.wandb_project:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or f"fold_{args.fold}",
            tags=[f"fold_{args.fold}"],
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model, log_freq=100)

    # Create trainer
    has_gpu = torch.cuda.is_available()
    accelerator = "gpu" if has_gpu else "cpu"
    devices = [int(args.devices)] if has_gpu else "auto"

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=loggers,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=30, verbose=True, mode="min")
        ],
        enable_progress_bar=True,
        deterministic=True,
    )

    # Train
    print("\nTraining...")
    trainer.fit(model, train_loader, val_loader)

    # Save model
    model_dir = MODELS_DIR / f"fold_{args.fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Evaluate on validation set - compute both overall and important-class metrics
    print("\nEvaluating on validation set...")
    model.eval()
    val_predictions = trainer.predict(model, val_loader)
    
    # Aggregate predictions
    all_pred_classes = torch.cat([batch["predictions"] for batch in val_predictions])
    all_targets = torch.cat([batch["targets"] for batch in val_predictions])
    
    # Overall accuracy
    overall_acc = (all_pred_classes == all_targets).float().mean().item()
    
    # Important-class accuracy (if using important loss, compute this)
    if args.use_important_loss:
        # Get important class indices
        label_to_idx = pd.Series(
            df["label_idx"].values, index=df["shortMergedCat"].values
        ).to_dict()
        important_indices = [
            label_to_idx[cat] for cat in IMPORTANT_CATEGORIES if cat in label_to_idx
        ]
        important_set = set(important_indices)
        
        # Filter to only samples where target is in important classes
        important_mask = torch.tensor(
            [t.item() in important_set for t in all_targets],
            dtype=torch.bool
        )
        
        if important_mask.any():
            important_acc = (
                all_pred_classes[important_mask] == all_targets[important_mask]
            ).float().mean().item()
            num_important_samples = important_mask.sum().item()
        else:
            important_acc = 0.0
            num_important_samples = 0
    else:
        important_acc = None
        num_important_samples = 0

    # Save config with metrics
    config_path = model_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Layer dims: {args.layer_dims}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Important categories: {args.use_important_loss}\n")
        f.write(f"Batch norm: {not args.no_batch_norm}\n")
        f.write(f"Layer norm: {args.use_layer_norm}\n")
        f.write(f"Residual: {not args.no_residual}\n")
        f.write(f"Attention: {args.use_attention}\n")
        f.write(f"Activation: {args.activation}\n")
        f.write(f"\nValidation Results:\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        if args.use_important_loss:
            f.write(f"Important-class accuracy: {important_acc:.4f}\n")
            f.write(f"Num important samples in validation: {num_important_samples}\n")

    print(f"\n✓ Results:")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    if args.use_important_loss:
        print(f"  Important-class accuracy: {important_acc:.4f} ({num_important_samples} samples)")


if __name__ == "__main__":
    main()
