# ASTMH Classification Pipeline

A streamlined, modular pipeline for training and evaluating abstract classifiers for the ASTMH conference.

## Overview

This pipeline simplifies the previous approach with ~5 core scripts:

1. **prepare_data**: Load Excel data, generate embeddings, save to parquet
2. **create_splits**: Create stratified 5-fold splits
3. **train**: Train classifier on each fold
4. **predict**: Make predictions on validation sets
5. **evaluate**: Aggregate results and generate metrics

## Directory Structure

```
modeling/
├── 1_prepare_data.py      # Load data & generate embeddings
├── 2_create_splits.py     # Create k-fold splits
├── 3_train.py             # Train classifier
├── 4_predict.py           # Make predictions
├── 5_evaluate.py          # Evaluate results
├── configs/
│   ├── config.py          # Configuration & paths
│   └── data_utils.py      # Data loading & processing
├── models/
│   └── classifier.py      # ASTMHClassifier model
├── loss/
│   └── custom_loss.py     # Custom loss for important categories
├── data/                  # Processed data & splits
├── logs/                  # Training logs
└── models/                # Saved model checkpoints
```

## Quick Start

### 1. Prepare Data

Load abstracts from Excel, generate embeddings using sentence-transformers, and save to parquet:

```bash
python 1_prepare_data.py
```

Options:
- `--force`: Regenerate embeddings even if output exists

**Output**: `data/embeddings_with_labels.parquet`

### 2. Create Splits

Generate stratified k-fold splits:

```bash
python 2_create_splits.py
```

Options:
- `--num_folds`: Number of folds (default: 5)

**Output**: `data/splits/fold_*_{train,val}.parquet`

### 3. Train Classifier

Train on each fold (repeat for each fold 0-4):

```bash
python 3_train.py --fold 0
```

Options:
- `--fold`: Fold number (required)
- `--epochs`: Number of epochs (default: 200)
- `--batch_size`: Batch size (default: 1024)
- `--lr`: Learning rate (default: 1e-4)
- `--layer_dims`: Hidden layer dimensions (default: 512 256)
- `--dropout`: Dropout probability (default: 0.4)
- `--use_important_loss`: Use custom loss for important categories
- `--wandb_project`: WandB project name (optional)
- `--wandb_name`: WandB run name (optional)
- `--devices`: GPU devices (default: "0")

**Output**: `models/fold_*/model.pt` and `models/fold_*/config.txt`

Examples:
```bash
# Basic training
python 3_train.py --fold 0

# With custom loss and WandB logging
python 3_train.py --fold 0 --use_important_loss --wandb_project MyProject --wandb_name fold_0

# Custom architecture
python 3_train.py --fold 0 --layer_dims 768 512 256 --lr 5e-5 --epochs 100
```

### 4. Make Predictions

Generate predictions on validation sets (repeat for each fold):

```bash
python 4_predict.py --fold 0
```

Options:
- `--fold`: Fold number (required)
- `--output_dir`: Output directory (default: results/)

**Output**: 
- `results/fold_*/predictions.parquet` - Detailed predictions
- `results/fold_*/summary.txt` - Fold-level summary

### 5. Evaluate Results

Aggregate predictions from all folds and compute metrics:

```bash
python 5_evaluate.py
```

Options:
- `--results_dir`: Results directory (default: results/)
- `--num_folds`: Number of folds (default: 5)

**Output**:
- `results/all_predictions.parquet` - Combined predictions
- `results/confusion_matrix.npy` - Confusion matrix
- `results/confusion_matrix.png` - Confusion matrix plot
- `results/summary.parquet` - Overall metrics

## Configuration

Edit `configs/config.py` to customize:

- **Paths**: Data directories, source file locations
- **Embedding model**: `EMBEDDING_MODEL` (default: 'all-mpnet-base-v2')
- **Training defaults**: Learning rate, batch size, layer dimensions
- **Important categories**: Categories to focus loss on
- **Splits**: Number of folds, random seed

## Custom Loss Function

The `ImportantCategoryLoss` in `loss/custom_loss.py` focuses training on ~29 priority categories (Malaria, Global Health, NTDs, Viruses). Samples where neither predicted nor target is important are treated as perfect predictions (excluded from loss).

Use with `--use_important_loss` flag in training:

```bash
python 3_train.py --fold 0 --use_important_loss
```

## Training Multiple Folds

Script to train all folds:

```bash
#!/bin/bash
for fold in {0..4}; do
    echo "Training fold $fold..."
    python 3_train.py --fold $fold --use_important_loss
    python 4_predict.py --fold $fold
done
echo "Evaluating results..."
python 5_evaluate.py
```

## Data Format

### Input (Excel)

Columns required:
- `title`: Abstract title
- `abstractText`: Full abstract text
- `shortMergedCat`: Category label

### Parquet Format

After step 1, embeddings are stored as:
- `emb_0`, `emb_1`, ..., `emb_767`: 768-dimensional embeddings
- `label_idx`: Encoded category index
- `shortMergedCat`: Category name
- Original columns: `title`, `abstractText`, etc.

## Prediction Output

`predictions.parquet` contains:
- `title`, `abstractText`, `shortMergedCat`, `label_idx`: Original data
- `pred_label_idx`: Predicted class index
- `pred_label`: Predicted class name
- `pred_confidence`: Model confidence for top prediction
- `correct`: Whether prediction matches ground truth
- `top2_pred_labels`, `top2_pred_confidence`: Top-2 predictions

## Logging with WandB

To log training runs to Weights & Biases:

```bash
python 3_train.py --fold 0 --wandb_project MyProject --wandb_name fold_0_baseline
```

## GPU Usage

Specify GPU devices:

```bash
# Single GPU
python 3_train.py --fold 0 --devices 0

# Multiple GPUs
python 3_train.py --fold 0 --devices 0,1
```

## Troubleshooting

**Embeddings not regenerating**: Use `--force` flag:
```bash
python 1_prepare_data.py --force
```

**Model not found during prediction**: Make sure training completed successfully:
```bash
ls models/fold_0/model.pt
```

**Memory issues**: Reduce batch size:
```bash
python 3_train.py --fold 0 --batch_size 256
```

**Slow training**: Reduce layer dimensions or use smaller dataset.
