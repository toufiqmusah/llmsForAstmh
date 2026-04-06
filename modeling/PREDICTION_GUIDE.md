# Prediction Pipeline Guide

## Workflow Overview

The pipeline has 5 steps. **Step 0 (embedding) is separate, then steps 1-5 are in sequence**:

```
Step 0: Embed Test Data (OUTSIDE PIPELINE)
        ↓
Step 1: Prepare & Embed Training Data
        ↓
Step 2: Create Stratified Splits
        ↓
Step 3: Train Classifiers
        ↓
Step 4: Make Predictions on Test Data ← NEW
        ↓
Step 5: Evaluate Results
```

## ✅ Should We Embed Test Data Outside?

**YES, absolutely.** Test embeddings should be generated BEFORE this pipeline:

1. **Test data should already be embedded** (using same embedding model as training)
2. **No labels needed for embeddings** (you can embed new data without knowing categories)
3. **Categories get assigned by predictions**

### How to Embed Test Data

```bash
# Before running the pipeline:
# Use the same sentence-transformer model to embed your test abstracts

# Option A: Use step 1_prepare_data.py with test data
python 1_prepare_data.py --input_file test_abstracts.xlsx \
                         --output_file data/test_embeddings.parquet

# Option B: Use external script to embed just test data
# See: ../pythonCode/abstracts2vec.py or similar
```

## Pipeline Steps Explained

### Step 0: Generate Test Embeddings (BEFORE PIPELINE)
```bash
# Embed test data using same model as training
# Input: test_abstracts.xlsx (needs 'abstractText' column, no categories needed)
# Output: test_embeddings.parquet (768-dim embeddings)
```

### Step 1: Prepare Training Data
```bash
python 1_prepare_data.py
# Input: Combined training spreadsheet with abstracts + categories
# Output: data/embeddings_with_labels.parquet (training embeddings + labels)
```

### Step 2: Create Stratified Splits
```bash
python 2_create_splits.py --num_folds 5
# Input: embeddings_with_labels.parquet
# Output: data/splits/fold_*_{train,val}.parquet
```

### Step 3: Train Classifiers
```bash
for fold in {0..4}; do
  python 3_train.py --fold $fold
done
# Input: fold-specific train/val splits
# Output: models/fold_*/model.pt
```

### Step 4: Make Predictions on Test Data (NEW)
```bash
# Option A: Predict with single fold
python 4_predict_test.py --test_data data/test_embeddings.parquet \
                         --fold 0 \
                         --output_dir results/

# Option B: Aggregate predictions from all folds (RECOMMENDED)
python 4_predict_test.py --test_data data/test_embeddings.parquet \
                         --all_folds \
                         --output_dir results/
```

**Output files:**
- `test_predictions.xlsx` - Detailed predictions with top 2 choices
- `confusion_matrix.png` - Visualization with jitter plot like old code
- `confusion_matrix.csv` - Confusion matrix as table
- `classification_report.txt` - Precision/recall per class
- `summary.txt` - Overall metrics

### Step 5: Evaluate Results
```bash
python 5_evaluate.py    # (original script, handles folds)
```

## Detailed Usage Examples

### 1. Quick Prediction on Test Data
```bash
cd modeling

# Assumes test_embeddings.parquet exists in data/
python 4_predict_test.py --fold 0 --output_dir results/test_fold0/

# Check results
cat results/test_fold0/summary.txt
# See: results/test_fold0/confusion_matrix.png
```

### 2. Best Practice: Aggregate All Folds
```bash
# More stable predictions using ensemble of all 5 folds
python 4_predict_test.py --all_folds \
                         --test_data data/test_embeddings.parquet \
                         --output_dir results/test_ensemble/

# Results use average probabilities across folds
```

### 3. Custom Test Path
```bash
# Predict on data at non-standard location
python 4_predict_test.py --test_data /path/to/my_test_embeddings.parquet \
                         --all_folds \
                         --output_dir results/custom_test/
```

## Output Files Explained

### test_predictions.xlsx
**Main results spreadsheet** - one row per test abstract:

| Column | Meaning |
|--------|---------|
| abstractId | Your abstract ID |
| given_Category | Original category (if known) |
| given_Category_idx | Original category as index |
| first_Pred_Category | Top prediction |
| first_Score | Confidence (0-100) |
| second_Pred_Category | 2nd choice (for review) |
| second_Score | 2nd confidence |
| correct | Did top prediction match? |
| title | Abstract title |
| abstractText | Abstract text |

### confusion_matrix.png
**Visualization like old code** with:
- Blue scatter points: predictions (true vs predicted)
- Green diagonal: perfect predictions
- Red lines: important category boundaries
- Jitter for visibility

### classification_report.txt
**Per-class metrics** (shows which categories are hard vs easy):
```
              precision    recall  f1-score   support
Malaria       0.82        0.85     0.83        100
Global Health 0.75        0.70     0.72         80
...
```

### summary.txt
**Quick metrics overview**:
```
Predictions from all folds
Test samples: 500
Overall accuracy: 0.7823
Important-class accuracy: 0.8456 (152 samples)
```

## How Predictions Work

### Single Fold
```python
# Load trained model for fold 0
model = ASTMHClassifier(...)
model.load(models/fold_0/model.pt)

# Get predictions
pred_classes = model(test_embeddings)  # Shape: (500,)
pred_probs = softmax(pred_classes)     # Shape: (500, 54)
```

### All Folds (Ensemble)
```python
# Average probabilities across all 5 models
probs_fold0 = model0(test_embeddings)  # (500, 54)
probs_fold1 = model1(test_embeddings)  # (500, 54)
...
probs_fold4 = model4(test_embeddings)  # (500, 54)

# Average
avg_probs = mean([probs_fold0, ..., probs_fold4])  # (500, 54)
predictions = argmax(avg_probs)                     # (500,)
```

**Why ensemble?** More stable, reduces overfitting to single fold.

## Test Data Format

**Input parquet file should have:**
```
Columns (required):
- 'embedding_0', 'embedding_1', ..., 'embedding_767'  (768-dim float embeddings)
- 'label_idx' (int, category index) - OPTIONAL for labeled test
- 'shortMergedCat' (str, category name) - OPTIONAL for labeled test

Columns (optional but useful):
- 'abstractId' - your identifier
- 'title' - abstract title
- 'abstractText' - full text
- Other metadata...
```

**If you don't have labels**, just provide embeddings. The script will still work:
```python
X_test = test_df[embedding_cols].values  # Just embeddings
# Rest works fine
```

## Troubleshooting

### Error: "Test data not found"
```bash
# Make sure test embeddings were generated
# Check path matches --test_data argument
python 4_predict_test.py --test_data data/test_embeddings.parquet --fold 0
```

### Error: "Model not found" for all folds
```bash
# Make sure models are trained first
python 3_train.py --fold 0
python 3_train.py --fold 1
# etc.
```

### Very low accuracy on test data
Possible reasons:
1. **Test set is very different** from training set
2. **Different embedding model** used for test vs train
3. **Label distribution** is very skewed
4. **Model is underfitted** - try different training hyperparameters

**Diagnostic:**
```bash
# Check accuracy on validation set (should be high)
cat modeling/models/fold_0/config.txt | grep accuracy

# Check test accuracy
cat results/test_ensemble/summary.txt
```

## Next Steps

1. **Generate test embeddings** (outside this pipeline, using same model)
2. **Run prediction**:
   ```bash
   cd modeling
   python 4_predict_test.py --all_folds --output_dir results/
   ```
3. **Review results** in `results/test_predictions.xlsx`
4. **Check confusion matrix** - `results/confusion_matrix.png`
5. **Manually review** low-confidence or incorrect predictions

## Code Reference

- [4_predict_test.py](4_predict_test.py) - New prediction script (like old predict.py)
- [3_train.py](3_train.py) - Training script (produces models/fold_*/model.pt)
- [models/classifier.py](models/classifier.py) - Neural network architecture
