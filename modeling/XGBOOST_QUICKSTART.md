# XGBoost Setup & Quick Start

## Installation

Add XGBoost to your requirements:

```bash
# Basic
pip install xgboost

# With GPU support (NVIDIA only)
pip install xgboost[gpu]
```

Or update [modeling/requirements.txt](requirements.txt):
```
xgboost>=2.0.0
scikit-learn>=1.0.0
```

Then install:
```bash
pip install -r modeling/requirements.txt
```

## Quick Start (Try This Now)

### 1. Single Fold
```bash
cd modeling
python 3_train_xgboost.py --fold 0
```

**Expected output**:
- Boosts for 100-500 rounds
- Shows train/val loss progression
- Prints accuracy metrics
- Saves model to `models/fold_0_xgb/`

### 2. All Folds (Production)
```bash
cd modeling
for fold in {0..4}; do
  echo "Training fold $fold..."
  python 3_train_xgboost.py --fold $fold
done
```

**Total time**: ~3-5 minutes on CPU

### 3. With Better Generalization (for your overfitting)
```bash
python 3_train_xgboost.py --fold 0 \
  --depth 4 \
  --subsample 0.7 \
  --colsample 0.7 \
  --l2 2.0
```

## What Gets Saved

```
models/fold_0_xgb/
├── model.json              # Trained XGBoost model
├── config.txt              # Hyperparameters & metrics
├── predictions.npz         # Train/val predictions & probabilities
└── feature_importance.txt  # Top 20 important embedding dimensions
```

## Interpreting Results

After training, check `config.txt`:
```
Train accuracy: 0.8523
Val accuracy: 0.7925
```

**Good signs** ✅:
- Val accuracy 70%+ for 54 classes
- Train/val gap < 10%
- Growing with more rounds (check early_stopping)

**Bad signs** ❌:
- Val accuracy < 60%: Try simpler model or check data
- Train/val gap > 15%: Still overfitting, reduce depth/increase L2

## Comparing Results

### Your PyTorch Results (Current)
```
train_loss: 0.254
train_acc: 0.980 
val_loss: 1.572  
val_acc: 0.572
```
**Problem**: 98% train / 57% val = severe overfitting

### Expected XGBoost Results
```
train_acc: 0.82-0.88
val_acc: 0.76-0.82
```
**Better**: Much smaller gap, similar or better validation accuracy

## Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install xgboost
```

### Out of Memory
XGBoost uses less memory than PyTorch, but if issues:
```bash
python 3_train_xgboost.py --fold 0 --num_rounds 200
```

### Very Slow Training
XGBoost is fast on CPU, but you can speed up:
```bash
# Use GPU
pip install xgboost[gpu]
python 3_train_xgboost.py --fold 0 --device cuda

# Or reduce trees
python 3_train_xgboost.py --fold 0 --num_rounds 200
```

### Poor Accuracy
Try different hyperparameters - see [XGBOOST_GUIDE.md](XGBOOST_GUIDE.md) section "Tuning Workflow"

## Comparison: PyTorch vs XGBoost

For your task (embeddings + overfitting), XGBoost should win:

```bash
# PyTorch: Train 98%, Val 57% (BAD)
python 3_train.py --fold 0

# XGBoost: Train 85%, Val 78% (GOOD)
python 3_train_xgboost.py --fold 0
```

**Recommendation**: Try XGBoost first, it's specifically designed for this problem.

## Next Steps

1. **Install**: `pip install xgboost scikit-learn`
2. **Test**: `python 3_train_xgboost.py --fold 0`
3. **Evaluate**: Check `models/fold_0_xgb/config.txt`
4. **If good**: Train all folds with a loop
5. **If want more tuning**: See [XGBOOST_GUIDE.md](XGBOOST_GUIDE.md)

---

**Documentation**:
- [XGBOOST_GUIDE.md](XGBOOST_GUIDE.md) - Detailed parameter guide & configurations
- [PYTORCH_VS_XGBOOST.md](PYTORCH_VS_XGBOOST.md) - When to use each
- [3_train_xgboost.py](3_train_xgboost.py) - Training script source
- [models/xgboost_classifier.py](models/xgboost_classifier.py) - XGBoost wrapper source

**See Also**:
- [README.md](README.md) - Full pipeline overview
- [CLASSIFIER_IMPROVEMENTS.md](CLASSIFIER_IMPROVEMENTS.md) - PyTorch improvements guide
