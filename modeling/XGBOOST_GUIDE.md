# XGBoost Classifier Guide

## Why XGBoost?

XGBoost is often **better than neural networks for tabular data** (like embeddings) because:

1. **Better Generalization** - Naturally resists overfitting with proper regularization
2. **Less Hyperparameter Tuning** - Good defaults work well
3. **Faster Training** - No GPU required, trains quickly on CPU
4. **Feature Importance** - Built-in understanding of which features matter
5. **Interpretability** - Decision trees are more interpretable than neural nets
6. **No Theory Needed** - Doesn't require understanding of deep learning

**Our Overfitting Problem**: Your neural net (train 98%, val 57%) suggests XGBoost would help.

## Quick Start

### Basic Training
```bash
python 3_train_xgboost.py --fold 0
```

Default parameters are well-tuned for classification. This should give you much better validation accuracy.

## Recommended Configurations

### 🎯 Baseline (Start Here)
```bash
python 3_train_xgboost.py --fold 0
```

**When to use**: First attempt, good for balanced data
**Typical result**: train 80-90%, val 75-85%

### 🚀 Better Generalization (for overfitting)
```bash
python 3_train_xgboost.py --fold 0 \
  --depth 4 \
  --subsample 0.7 \
  --colsample 0.7 \
  --l2 2.0
```

**When to use**: If you see overfitting (train >> val)
**Changes**:
- Shallower trees (depth 4 vs 6) = simpler model
- Lower subsample (70% vs 80%) = more noise/regularization
- Lower colsample (70% vs 80%) = fewer features per tree
- Higher L2 (2.0 vs 1.0) = stronger regularization

**Expected result**: train 75-85%, val 75-82% (balanced)

### ⚡ Lightweight (Fast Training)
```bash
python 3_train_xgboost.py --fold 0 \
  --depth 4 \
  --num_rounds 200 \
  --early_stopping 30
```

**When to use**: Quick experiments, limited compute time
**Time**: ~5-10s vs 30-60s for baseline

### 💪 More Complex (Deeper Model)
```bash
python 3_train_xgboost.py --fold 0 \
  --depth 8 \
  --subsample 0.9 \
  --colsample 0.9 \
  --lr 0.05
```

**When to use**: If baseline underfits (train and val both low)
**Changes**:
- Deeper trees (8 vs 6) = higher capacity
- Higher subsample/colsample = more stable
- Lower learning rate = finer optimization

### 🔬 Aggressive Regularization (Maximum Generalization)
```bash
python 3_train_xgboost.py --fold 0 \
  --depth 3 \
  --subsample 0.6 \
  --colsample 0.6 \
  --l2 5.0 \
  --min_child_weight 3
```

**When to use**: Data is imbalanced or classes are very hard to separate
**Expected**: Simpler model, better validation stability

## Parameter Explanations

### `--depth` (Max Tree Depth)
- **Range**: 3-10
- **Default**: 6
- **Impact**: Higher = more complex, slower, more overfitting
- **Tuning**: Start at 6, go down if overfitting, up if underfitting

```
depth=3: Simple, fast, may underfit
depth=6: Balanced (default)
depth=8: Complex, slower, more overfitting risk
```

### `--lr` (Learning Rate)
- **Range**: 0.01-0.5
- **Default**: 0.1
- **Impact**: Lower = slower learning, more stable; Higher = faster, more noise
- **Tuning**: Try 0.05-0.2 range

```
lr=0.01: Very slow but stable
lr=0.1:  Balanced (default)
lr=0.3:  Fast but may oscillate
```

### `--subsample` (Row Sampling)
- **Range**: 0.5-1.0
- **Default**: 0.8
- **Impact**: Lower = less overfitting, but slower convergence
- **Tuning**: 0.7-0.9 for regularization

```
subsample=0.7: Strong regularization (less overfitting)
subsample=0.8: Balanced (default)
subsample=0.95: Minimal regularization
```

### `--colsample` (Column/Feature Sampling)
- **Range**: 0.5-1.0
- **Default**: 0.8
- **Impact**: Similar to subsample but for features
- **Tuning**: Lower if overfitting, higher if underfitting

```
colsample=0.6: Strong regularization
colsample=0.8: Balanced (default)
colsample=1.0: Use all features
```

### `--l2` (L2 Regularization)
- **Range**: 0.0-10.0
- **Default**: 1.0
- **Impact**: Higher = simpler trees, less overfitting
- **Tuning**: Try 0.5-3.0 for overfitting

```
l2=0.5:  Light regularization
l2=1.0:  Balanced (default)
l2=3.0:  Strong regularization
l2=5.0:  Very strong regularization
```

### `--min_child_weight` (Minimum Leaf Weight)
- **Range**: 1-10
- **Default**: 1
- **Impact**: Higher = less overfitting, simpler trees
- **Tuning**: Try 1-3 for regularization

```
min_child_weight=1: Minimal constraint
min_child_weight=3: Add regularization
min_child_weight=5: Strong constraint
```

### `--num_rounds` (Boosting Rounds)
- **Range**: 100-2000
- **Default**: 500
- **Impact**: More rounds = more complex model
- **Note**: Early stopping will stop before this

```
num_rounds=200:  Fast training
num_rounds=500:  Balanced (default)
num_rounds=1000: More capacity
```

### `--early_stopping` (Patience)
- **Range**: 10-100
- **Default**: 50
- **Impact**: Stops if validation doesn't improve for N rounds
- **Tuning**: Usually leave at default (50)

## Tuning Workflow

### Step 1: Baseline
```bash
python 3_train_xgboost.py --fold 0
# Check: train_acc vs val_acc gap
```

### Step 2a: If OVERFITTING (train >> val)
```bash
# Reduce model complexity
python 3_train_xgboost.py --fold 0 \
  --depth 4 \
  --subsample 0.7 \
  --colsample 0.7 \
  --l2 2.0
```

### Step 2b: If UNDERFITTING (train and val both low)
```bash
# Increase model complexity
python 3_train_xgboost.py --fold 0 \
  --depth 8 \
  --subsample 0.9 \
  --colsample 0.9 \
  --lr 0.05
```

### Step 3: Fine-tune
Adjust parameters incrementally based on results.

## Expected Results

Typical accuracy ranges for 54-class classification on ASTMH data:

| Configuration | Train | Val | Interpretation |
|--------------|-------|-----|-----------------|
| Shallow, regularized | 80-85% | 78-82% | Good generalization ✅ |
| Baseline | 85-90% | 80-85% | Balanced ✅ |
| Deep | 90-95% | 80-82% | Some overfitting ⚠️ |
| Very deep | 98%+ | 70-75% | Severe overfitting ❌ |

**Goal**: Minimize the gap between train and validation accuracy.

## Comparison: XGBoost vs Neural Network

For your overfitting issue:

| Aspect | XGBoost | Neural Net |
|--------|---------|-----------|
| Generalization | Better (natural regularization) | Needs careful tuning |
| Training Speed | Fast (CPU-friendly) | Slower (GPU needed) |
| Hyperparameter Tuning | Easier, fewer parameters | Hard, many parameters |
| Interpretability | Feature importance built-in | Black box |
| Best for | Tabular/embedding data | Images, sequences |
| Memory | Low | High (GPU) |

**Verdict**: For your embeddings + overfitting issue, **XGBoost is likely better**.

## Advanced: Feature Importance

After training, check `feature_importance.txt` to see which embedding dimensions matter most:

```
Feature importance (top 20):
  f_123: 45
  f_456: 42
  f_789: 38
  ...
```

High-importance features = dimensions most predictive for classification.

## Troubleshooting

### Training too slow?
- Reduce `--num_rounds` (200-300)
- Reduce `--depth` (4-5)
- Use `--early_stopping 20` for faster stopping

### Overfitting still present?
- Lower `--subsample` (0.5-0.6)
- Lower `--colsample` (0.5-0.6)
- Increase `--l2` (3.0-5.0)
- Increase `--min_child_weight` (2-5)

### Val accuracy very low?
- Check if data is loaded correctly
- Increase `--depth` (7-8)
- Increase `--lr` (0.2)
- Increase `--num_rounds` (800-1000)

## Next Steps

1. **Run baseline**: `python 3_train_xgboost.py --fold 0`
2. **Compare with neural net**: Same fold, compare val_acc
3. **Tune if needed**: Use overfitting config if baseline shows overfitting
4. **Train all folds**: Loop through folds 0-4
5. **Evaluate**: Aggregate results like before

---

**Installation**: Make sure xgboost is installed:
```bash
pip install xgboost
```

**GPU Support** (optional but faster):
```bash
# NVIDIA GPU
pip install xgboost[gpu]
```

**Time to train per fold**: ~30-60 seconds (CPU) or ~5-15 seconds (GPU)

---

**See Also**:
- [3_train_xgboost.py](3_train_xgboost.py) - Training script
- [models/xgboost_classifier.py](models/xgboost_classifier.py) - Implementation
- [CLASSIFIER_IMPROVEMENTS.md](CLASSIFIER_IMPROVEMENTS.md) - Neural network tuning guide
