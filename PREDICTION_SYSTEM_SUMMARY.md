# New Prediction System - Summary

## What Was Added

### 🆕 New File: [4_predict_test.py](modeling/4_predict_test.py)

Adapted from your old `pythonCode/predict.py` but integrated with the new pipeline.

**Key Features:**
- ✅ Works with trained models from step 3
- ✅ Generates predictions with top-2 choices and confidence scores
- ✅ Creates confusion matrix visualization (jittered, like old code)
- ✅ Saves results to Excel for easy review
- ✅ Can use single fold or aggregate all 5 folds
- ✅ Reports both overall and important-class accuracy

**Usage:**
```bash
# Best practice: aggregate all 5 folds
python 4_predict_test.py --all_folds \
                         --test_data data/test_embeddings.parquet \
                         --output_dir results/test/

# Or single fold
python 4_predict_test.py --fold 0 \
                         --test_data data/test_embeddings.parquet \
                         --output_dir results/test/
```

### 📖 New Guide: [PREDICTION_GUIDE.md](modeling/PREDICTION_GUIDE.md)

Comprehensive workflow guide covering:
- Complete 5-step pipeline (Step 0 is embedding)
- Why test embeddings should be pre-generated
- How to embed test data
- Detailed usage examples
- Output file explanations
- Troubleshooting tips

### 📝 Updated: [README.md](modeling/README.md)

Added section 4 for test predictions with:
- Usage examples
- Output descriptions
- Links to guides

---

## Answer: Should We Embed Test Data Outside?

### ✅ YES - Absolutely

**Workflow:**
```
Step 0 (OUTSIDE PIPELINE): Embed test abstracts
        ↓
        Generate test_embeddings.parquet using same embedding model
        
Then run pipeline normally with trained models
```

**Why?**
1. **Test data may not have categories** - You're predicting them!
2. **Same embedding model needed** - Must match training
3. **Cleaner separation of concerns** - Embedding ≠ prediction

**How to embed:**
```bash
# Use same sentence-transformer model as training
# Input: test_abstracts.xlsx (just abstractText + abstractId needed)
# Output: test_embeddings.parquet (768-dim vectors)

# Option A: Modify step 1 for test data
python 1_prepare_data.py --input_file test_abstracts.xlsx \
                         --output_file data/test_embeddings.parquet

# Option B: Use existing embed script
python ../pythonCode/abstracts2vec.py --input test_abstracts.xlsx \
                                      --output data/test_embeddings.parquet
```

---

## How It Compares to Old Code

| Feature | Old Code | New Code |
|---------|----------|----------|
| **Checkpoint loading** | PyTorch Lightning | Standard PyTorch state_dict |
| **Data loading** | Custom EmbeddingData | Simple pandas parquet |
| **Top-2 predictions** | Manual logic | Built-in function |
| **Confusion matrix** | Custom scatter plot | Recreated exactly like old |
| **Multi-fold** | Manual per-fold | Automated ensemble |
| **Output format** | Single xlsx | Excel + PNG + CSV + TXT |
| **Accuracy metrics** | Basic | Overall + Important-class |

---

## Output Files

After running `4_predict_test.py`:

```
results/test/
├── test_predictions.xlsx       # Main results (like old code)
├── confusion_matrix.png        # Visualization with jitter
├── confusion_matrix.csv        # Numerical matrix
├── classification_report.txt   # Per-class metrics
└── summary.txt                 # Quick metrics
```

### test_predictions.xlsx Columns

```
abstractId              | unique ID from your data
given_Category          | original category (if known)
first_Pred_Category     | top prediction (NEW)
first_Score             | confidence 0-100 (NEW)
second_Pred_Category    | 2nd choice (NEW)
second_Score            | 2nd confidence (NEW)
correct                 | did top match? (NEW)
title                   | abstract title
abstractText            | full abstract
```

### confusion_matrix.png

Visualization exactly like old code:
- **Blue dots**: predictions (true vs pred, jittered)
- **Green diagonal**: perfect predictions
- **Red lines**: important category boundaries
- **Top/side markers**: highlighted category distribution

---

## Quick Start

### 1. Embed test data (outside pipeline)
```bash
# Assuming test abstracts in data/test_abstracts.xlsx
python ../pythonCode/abstracts2vec.py \
  --input data/test_abstracts.xlsx \
  --output data/test_embeddings.parquet
```

### 2. Run prediction (inside pipeline)
```bash
cd modeling
python 4_predict_test.py --all_folds \
                         --test_data ../data/test_embeddings.parquet \
                         --output_dir results/test/
```

### 3. Review results
```bash
# Open Excel for predictions
open results/test/test_predictions.xlsx

# View confusion matrix
open results/test/confusion_matrix.png

# Check metrics
cat results/test/summary.txt
```

---

## Files Modified/Created

**New:**
- [modeling/4_predict_test.py](modeling/4_predict_test.py) - Prediction script
- [modeling/PREDICTION_GUIDE.md](modeling/PREDICTION_GUIDE.md) - Workflow guide

**Updated:**
- [modeling/README.md](modeling/README.md) - Added step 4 documentation

---

## Using Both Models (PyTorch + XGBoost)

If you trained both, you can make predictions with both and ensemble:

```bash
# PyTorch predictions (all folds)
python 4_predict_test.py --all_folds \
                         --output_dir results/pytorch/

# XGBoost predictions (different script, different models)
# Would need equivalent 4_predict_test_xgboost.py
# (Can create if needed)

# Then ensemble average in spreadsheet
```

---

**Ready to use!** Test embeddings → predictions → confusion matrix → Excel results.
