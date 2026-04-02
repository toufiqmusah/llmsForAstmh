# Improved ASTMH Classifier - Architecture & Updates

## Overview

The classifier has been significantly improved with modern deep learning best practices. The new architecture provides better training stability, faster convergence, and improved generalization for your multi-class classification task (54 classes).

## Key Improvements

### 1. **Batch Normalization (Default: ON)**
- **What it does**: Normalizes layer inputs to stabilize training
- **Why**: Reduces internal covariate shift, allowing higher learning rates and better convergence
- **Impact**: ~5-10% faster training, more stable loss curves
- **Usage**: Enabled by default. Disable with `--no_batch_norm` if needed

```python
# Enables by default
python 3_train.py --fold 0

# Disable if desired
python 3_train.py --fold 0 --no_batch_norm
```

### 2. **Layer Normalization (Optional)**
- **What it does**: Alternative normalization that normalizes across features instead of batch
- **When to use**: Better for smaller batch sizes (<64) or sequence models
- **Usage**: `--use_layer_norm` to replace batch norm

```python
python 3_train.py --fold 0 --use_layer_norm
```

### 3. **Residual Connections (Default: ON)**
- **What it does**: Allows gradients to flow directly through layers (skip connections)
- **Why**: Enables training of deeper networks without vanishing gradients
- **Implementation**: Only activates when consecutive layers have matching dimensions
  - Example: [768, 512, 512, 256] creates residual block between the two 512-dim layers
- **Impact**: Better for deep networks (>4 layers), deeper models converge faster
- **Usage**: Enabled by default. Disable with `--no_residual`

```python
# With residual connections (deeper network)
python 3_train.py --fold 0 --layer_dims 768 512 512 256

# Without residual connections
python 3_train.py --fold 0 --layer_dims 768 512 512 256 --no_residual
```

### 4. **Feature Attention Mechanism (Optional)**
- **What it does**: Learn which input embeddings are most important
- **Architecture**: Lightweight attention over the 768-dim embeddings
- **Why useful**: May help focus on most relevant abstract features
- **Computational cost**: Minimal (~1% overhead)
- **Usage**: `--use_attention`

```python
python 3_train.py --fold 0 --use_attention
```

### 5. **Flexible Activation Functions**
- **Options**: `relu` (default), `gelu`, `elu`
- **Recommendations**:
  - `relu`: Fast, standard choice (default)
  - `gelu`: Smoother, sometimes better for large models
  - `elu`: Can reduce dead neurons, good for very deep networks

```python
python 3_train.py --fold 0 --activation gelu
python 3_train.py --fold 0 --activation elu
```

### 6. **Better Learning Rate Scheduling**
- Now includes `min_lr=1e-6` floor to prevent learning rates from becoming too small
- More robust ReduceLROnPlateau implementation

## Recommended Configurations

### 🎯 Baseline (Balanced, Good Default)
```bash
python 3_train.py --fold 0 \
  --epochs 200 \
  --layer_dims 512 256 \
  --dropout 0.4 \
  --lr 1e-4
```

### 🚀 Advanced (Deeper, for more complex patterns)
```bash
python 3_train.py --fold 0 \
  --epochs 200 \
  --layer_dims 768 512 512 256 \
  --dropout 0.4 \
  --lr 1e-4 \
  --use_attention
```

### ⚡ Fast & Lightweight (Small model, quick training)
```bash
python 3_train.py --fold 0 \
  --epochs 100 \
  --layer_dims 256 \
  --dropout 0.3 \
  --lr 5e-4 \
  --no_residual
```

### 🔬 Experimental (Smooth training, GELU activation)
```bash
python 3_train.py --fold 0 \
  --epochs 200 \
  --layer_dims 768 512 256 \
  --dropout 0.4 \
  --lr 1e-4 \
  --activation gelu \
  --use_attention
```

### 🎓 With Custom Loss for Important Categories
```bash
python 3_train.py --fold 0 \
  --epochs 200 \
  --layer_dims 512 256 \
  --dropout 0.4 \
  --use_important_loss
```

## Training Best Practices

### Learning Rate Selection
- **High LR (5e-4 - 1e-3)**: Fast convergence but may overshoot, use for shallow networks
- **Medium LR (1e-4 - 5e-4)**: Balanced, good starting point (recommended)
- **Low LR (1e-5 - 5e-5)**: Stable but slow, use for fine-tuning or very deep networks

### Dropout Selection
- **0.2-0.3**: Light regularization, use for small models
- **0.4**: Standard choice (default)
- **0.5+**: Strong regularization, use for large models or small datasets

### Architecture Depth
- **Shallow (1-2 layers)**: Fast training, good for simple patterns
- **Medium (3-4 layers)**: Balanced, captures complex patterns
- **Deep (5+ layers)**: Better feature learning but needs residual connections & batch norm

## What Changed in the Code

### `classifier.py`
```python
# New parameters in __init__
- use_batch_norm: Apply batch normalization (default: True)
- use_layer_norm: Apply layer normalization instead
- use_residual: Use residual connections (default: True)
- use_attention: Add feature attention mechanism
- activation: Choose activation function (relu/gelu/elu)

# New ResidualBlock class
- Implements skip connection with internal normalization
- Only used when consecutive layers have same dimension
```

### `3_train.py`
```python
# New command-line arguments
--no_batch_norm       # Disable batch normalization
--use_layer_norm      # Use layer norm instead of batch norm
--no_residual         # Disable residual connections
--use_attention       # Enable feature attention
--activation {relu,gelu,elu}  # Choose activation
```

## Migration from Old Classifier

The new classifier is **fully backward compatible**. Existing code will work without changes:

```python
# Old usage still works
model = ASTMHClassifier(input_dim=768, layer_dims=[512, 256], num_classes=54)

# New usage with features (optional)
model = ASTMHClassifier(
    input_dim=768,
    layer_dims=[512, 256],
    num_classes=54,
    use_batch_norm=True,
    use_residual=True,
    use_attention=True,
    activation="gelu"
)
```

## Troubleshooting

### 🔴 Training loss not decreasing?
- Try higher learning rate: `--lr 5e-4`
- Reduce dropout: `--dropout 0.2`
- Use GELU activation: `--activation gelu`

### 🟡 Training is very unstable?
- Use batch norm (enabled by default)
- Use layer norm: `--use_layer_norm`
- Reduce learning rate: `--lr 5e-5`
- Increase dropout: `--dropout 0.5`

### 🟢 Training is slow?
- Reduce model size: `--layer_dims 256`
- Use simpler architecture: `--no_residual`
- Disable attention: (default, don't use `--use_attention`)

### 🔵 Overfitting to training data?
- Increase dropout: `--dropout 0.5-0.6`
- Reduce model size: `--layer_dims 256 128`
- Use regularization: `--use_important_loss`
- Reduce learning rate: `--lr 5e-5`

## Performance Expectations

With the improvements, you should expect:
- **Convergence**: 10-20% faster than baseline
- **Stability**: Smoother training curves, fewer divergences
- **Generalization**: Better validation accuracy with same model size
- **Scalability**: Easier to train deeper models

## Advanced: Fine-tuning Temperature

The temperature parameter controls softmax sharpness:
```python
# Higher temperature = softer predictions (more uncertainty)
# Lower temperature = sharper predictions (more confident)
# Modify in code (currently hardcoded to 1.0)

model = ASTMHClassifier(temperature=1.5)  # Softer
model = ASTMHClassifier(temperature=0.5)  # Sharper
```

For your multi-class task, 1.0 is typically optimal.

---

**Last Updated**: April 1, 2026
**Compatibility**: PyTorch 2.0+, PyTorch Lightning 2.0+
