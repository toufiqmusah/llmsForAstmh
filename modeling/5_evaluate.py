"""
5_evaluate.py

Aggregate predictions from all folds, compute metrics, and generate visualizations.

Usage:
    python 5_evaluate.py [--results_dir RESULTS] [--num_folds 5]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions across all folds"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Results directory",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 5: EVALUATE & GENERATE RESULTS")
    print("=" * 80)

    # Load predictions from all folds
    all_predictions = []
    fold_accuracies = []

    for fold in range(args.num_folds):
        fold_dir = args.results_dir / f"fold_{fold}"
        pred_file = fold_dir / "predictions.parquet"

        if not pred_file.exists():
            print(f"Warning: Predictions for fold {fold} not found")
            continue

        df = pd.read_parquet(pred_file)
        all_predictions.append(df)

        accuracy = np.mean(df["correct"])
        fold_accuracies.append(accuracy)
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}")

    if not all_predictions:
        print("ERROR: No predictions found!")
        return

    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)

    # Overall metrics
    overall_accuracy = np.mean(combined_df["correct"])
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Per-class metrics
    target_labels = combined_df["shortMergedCat"].unique()
    
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}

    print("\nPer-class metrics:")
    print(f"{'Class':<40} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 76)

    for label in sorted(target_labels):
        mask = combined_df["shortMergedCat"] == label
        if mask.sum() == 0:
            continue

        subset = combined_df[mask]
        precision = (subset["correct"]).sum() / len(subset)
        recall = (subset["correct"]).sum() / len(subset)
        
        precision_per_class[label] = precision
        recall_per_class[label] = recall
        f1_per_class[label] = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(
            f"{label:<40} {precision:<12.4f} {recall:<12.4f} {f1_per_class[label]:<12.4f}"
        )

    # Save confusion matrix
    y_true = combined_df["label_idx"].values
    y_pred = combined_df["pred_label_idx"].values
    cm = confusion_matrix(y_true, y_pred)

    cm_path = args.results_dir / "confusion_matrix.npy"
    np.save(cm_path, cm)
    print(f"\n✓ Confusion matrix saved to {cm_path}")

    # Plot confusion matrix (if not too large)
    if len(target_labels) <= 20:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_plot_path = args.results_dir / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrix plot saved to {cm_plot_path}")
        plt.close()

    # Save overall results
    summary_df = pd.DataFrame({
        "Metric": ["Overall Accuracy", "Mean Fold Accuracy", "Std Fold Accuracy"],
        "Value": [overall_accuracy, np.mean(fold_accuracies), np.std(fold_accuracies)],
    })

    results_path = args.results_dir / "summary.parquet"
    summary_df.to_parquet(results_path)
    print(f"\n✓ Summary saved to {results_path}")

    # Save combined predictions
    combined_path = args.results_dir / "all_predictions.parquet"
    combined_df.to_parquet(combined_path)
    print(f"✓ Combined predictions saved to {combined_path}")

    print("\n" + "=" * 80)
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
