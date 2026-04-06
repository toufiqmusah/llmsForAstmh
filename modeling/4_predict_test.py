"""
4_predict_test.py

Make predictions on test data using trained classifiers.

Follows the workflow from the old pythonCode/predict.py but adapted for the new pipeline.

Usage:
    python 4_predict_test.py --test_data path/to/test_embeddings.parquet \
                             --fold 0 \
                             --output_dir results/
                             
    # Or predict with all folds and aggregate
    python 4_predict_test.py --test_data path/to/test_embeddings.parquet \
                             --all_folds \
                             --output_dir results/

Assumes:
1. Test embeddings have already been generated (step 1_prepare_data.py)
2. Models are trained for desired folds (step 3_train.py)
3. Test data has 'label_idx' and 'shortMergedCat' columns
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from tqdm import tqdm

from configs.config import (
    EMBEDDINGS_PARQUET,
    SPLITS_DIR,
    MODELS_DIR,
    IMPORTANT_CATEGORIES,
)
from configs.data_utils import get_embedding_columns
from models.classifier import ASTMHClassifier


def load_model(model_path, input_dim, num_classes, device):
    """Load trained PyTorch model."""
    model = ASTMHClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        layer_dims=[512, 256],  # Default, doesn't matter for inference
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_category_ordering(train_df):
    """Get alphabetical ordering of categories as used by model."""
    # Categories are sorted alphabetically
    cat_to_idx = pd.Series(
        train_df["label_idx"].values, index=train_df["shortMergedCat"].values
    ).to_dict()
    idx_to_cat = {v: k for k, v in cat_to_idx.items()}
    sorted_indices = sorted(idx_to_cat.keys())
    sorted_cats = [idx_to_cat[i] for i in sorted_indices]
    return sorted_cats, cat_to_idx


def make_predictions(model, X, device, batch_size=1024):
    """
    Get predictions from model.
    
    Returns:
        pred_classes: argmax predictions
        pred_probs: full probability distributions
    """
    pred_classes = []
    pred_probs = []
    
    num_batches = (len(X) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            X_batch = X[i * batch_size : (i + 1) * batch_size]
            X_tensor = torch.FloatTensor(X_batch).to(device)
            
            # Forward pass gets log-softmax
            log_probs = model.forward(X_tensor)
            probs = torch.exp(log_probs)
            
            # Get predictions
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            batch_probs = probs.cpu().numpy()
            
            pred_classes.extend(batch_preds)
            pred_probs.extend(batch_probs)
    
    return np.array(pred_classes), np.array(pred_probs)


def plot_confusion_matrix(
    true_labels,
    pred_labels,
    sorted_cats,
    figsize=(20, 20),
    title="Confusion Matrix"
):
    """
    Create confusion matrix plot like the old code.
    
    Includes:
    - Jittered scatter plot of predictions
    - Green diagonal (perfect predictions)
    - Red lines delineating important categories
    - Top/side axes showing tested categories
    """
    num_classes = len(sorted_cats)
    
    # Get importance from config
    important_cats = set(IMPORTANT_CATEGORIES)
    important_indices = [
        i for i, cat in enumerate(sorted_cats) if cat in important_cats
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Jitter for visibility
    jit_x = -0.3 + 0.6 * np.random.random(len(true_labels))
    jit_y = -0.3 + 0.6 * np.random.random(len(true_labels))
    
    # Plot predictions
    ax.scatter(
        true_labels + jit_x,
        pred_labels + jit_y,
        alpha=0.5,
        s=20,
        c='blue',
        label='Predictions'
    )
    
    # Green diagonal (perfect predictions)
    ax.plot([0, num_classes], [0, num_classes], 'g:', linewidth=2, label='Perfect')
    
    # Red lines for important category boundaries
    if important_indices:
        # Get boundaries
        boundaries = [i + 0.5 for i in important_indices if i < num_classes - 1]
        for b in boundaries:
            ax.hlines(y=b, xmin=-1, xmax=num_classes, colors='r', alpha=0.3)
            ax.vlines(x=b, ymin=-1, ymax=num_classes, colors='r', alpha=0.3)
    
    # Axes
    ax.set_xlim((-1, num_classes))
    ax.set_ylim((-1, num_classes))
    ax.set_xlabel('True Category', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Category', fontweight='bold', fontsize=12)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    
    # Shorter labels for readability
    short_labels = [cat.replace(' - ', '\n')[:20] for cat in sorted_cats]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Make predictions on test data")
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test embeddings parquet file. If None, expects path from config.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to use. If None with --all_folds, uses all folds.",
    )
    parser.add_argument(
        "--all_folds",
        action="store_true",
        help="Use all folds and aggregate predictions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.fold is None and not args.all_folds:
        raise ValueError("Must specify either --fold or --all_folds")

    print("=" * 80)
    print("STEP 4: PREDICT ON TEST DATA")
    print("=" * 80)

    # Load test data
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        # Assume test is same format as training embeddings
        test_path = Path("data/test_embeddings.parquet")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_parquet(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Get embedding columns
    embedding_cols = get_embedding_columns(test_df)
    print(f"Embedding dimensions: {len(embedding_cols)}")

    # Load training data to get category ordering
    print(f"Loading training data for category ordering...")
    train_df = pd.read_parquet(EMBEDDINGS_PARQUET)
    sorted_cats, cat_to_idx = get_category_ordering(train_df)
    num_classes = len(sorted_cats)
    print(f"Number of classes: {num_classes}")
    
    # Get important category indices
    important_set = set([
        cat_to_idx[cat] for cat in IMPORTANT_CATEGORIES if cat in cat_to_idx
    ])

    # Prepare test features and labels
    X_test = test_df[embedding_cols].values
    y_test = test_df["label_idx"].values
    test_cats = test_df["shortMergedCat"].values
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine which folds to use
    if args.all_folds:
        folds_to_use = list(range(5))
        print(f"Using all folds: {folds_to_use}")
    else:
        folds_to_use = [args.fold]
        print(f"Using fold: {args.fold}")

    # Make predictions for each fold and aggregate
    all_pred_classes = []
    all_pred_probs = []
    
    for fold in folds_to_use:
        print(f"\n--- Fold {fold} ---")
        
        model_path = MODELS_DIR / f"fold_{fold}" / "model.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}, skipping")
            continue
        
        # Load model
        model = load_model(model_path, len(embedding_cols), num_classes, device)
        print(f"Loaded model from {model_path}")
        
        # Make predictions
        print("Making predictions...")
        pred_classes, pred_probs = make_predictions(model, X_test, device)
        
        all_pred_classes.append(pred_classes)
        all_pred_probs.append(pred_probs)
    
    # Average predictions across folds if multiple
    if len(all_pred_probs) > 1:
        print(f"\nAggregating predictions from {len(all_pred_probs)} folds...")
        avg_probs = np.mean(all_pred_probs, axis=0)
        pred_classes = np.argmax(avg_probs, axis=1)
        pred_probs = avg_probs
    else:
        pred_classes = all_pred_classes[0]
        pred_probs = all_pred_probs[0]

    # Get top 2 predictions for each sample
    print("\nProcessing predictions...")
    first_pred_classes = pred_classes
    first_pred_cats = np.array([sorted_cats[i] for i in first_pred_classes])
    first_scores = np.max(pred_probs, axis=1)
    
    # Get second best prediction
    second_pred_probs = pred_probs.copy()
    second_pred_probs[np.arange(len(pred_probs)), first_pred_classes] = -1
    second_pred_classes = np.argmax(second_pred_probs, axis=1)
    second_pred_cats = np.array([sorted_cats[i] for i in second_pred_classes])
    second_scores = np.max(second_pred_probs, axis=1)

    # Compute metrics
    overall_acc = (first_pred_classes == y_test).mean()
    
    # Accuracy on important classes only
    important_mask = np.array([y in important_set for y in y_test])
    if important_mask.any():
        important_acc = (
            first_pred_classes[important_mask] == y_test[important_mask]
        ).mean()
        num_important = important_mask.sum()
    else:
        important_acc = 0.0
        num_important = 0

    print(f"\n✓ Results:")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Important-class accuracy: {important_acc:.4f} ({num_important} samples)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    print(f"\nGenerating confusion matrix...")
    fig = plot_confusion_matrix(
        y_test,
        first_pred_classes,
        sorted_cats,
        title="Confusion Matrix - Test Set"
    )
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close(fig)

    # Save results to Excel
    print(f"Saving predictions to Excel...")
    results_df = pd.DataFrame({
        'abstractId': test_df.get('abstractId', range(len(test_df))).values,
        'given_Category': test_cats,
        'given_Category_idx': y_test,
        'first_Pred_Category': first_pred_cats,
        'first_Pred_Category_idx': first_pred_classes,
        'first_Score': np.round(first_scores * 100, 2),
        'second_Pred_Category': second_pred_cats,
        'second_Pred_Category_idx': second_pred_classes,
        'second_Score': np.round(second_scores * 100, 2),
        'correct': first_pred_classes == y_test,
        'title': test_df.get('title', [''] * len(test_df)).values,
        'abstractText': test_df.get('abstractText', [''] * len(test_df)).values,
    })

    results_path = output_dir / "test_predictions.xlsx"
    results_df.to_excel(results_path, index=False)
    print(f"✓ Predictions saved to {results_path}")

    # Save classification report
    report = classification_report(y_test, first_pred_classes, 
                                   target_names=sorted_cats, 
                                   zero_division=0)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Classification report saved to {report_path}")

    # Save confusion matrix as CSV
    cm = confusion_matrix(y_test, first_pred_classes, labels=range(num_classes))
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df = pd.DataFrame(cm, index=sorted_cats, columns=sorted_cats)
    cm_df.to_csv(cm_path)
    print(f"✓ Confusion matrix (CSV) saved to {cm_path}")

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        if args.all_folds:
            f.write(f"Predictions aggregated from folds: {folds_to_use}\n")
        else:
            f.write(f"Predictions from fold: {args.fold}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"Important-class accuracy: {important_acc:.4f} ({num_important} samples)\n")
    print(f"✓ Summary saved to {summary_path}")

    print(f"\n✓ All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
