#!/bin/bash

# run_pipeline.sh
# Complete pipeline runner with customizable options

set -e  # Exit on error

# Default values
NUM_FOLDS=5
USE_IMPORTANT_LOSS=false
WANDB_PROJECT=""
FORCE_EMBEDDINGS=false
DEVICE="0"
BATCH_SIZE=1024
LR=1e-4
EPOCHS=200
LAYER_DIMS="512 256"
DROPOUT=0.4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_folds)
            NUM_FOLDS=$2
            shift 2
            ;;
        --use_important_loss)
            USE_IMPORTANT_LOSS=true
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT=$2
            shift 2
            ;;
        --force_embeddings)
            FORCE_EMBEDDINGS=true
            shift
            ;;
        --device)
            DEVICE=$2
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --lr)
            LR=$2
            shift 2
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --layer_dims)
            LAYER_DIMS=$2
            shift 2
            ;;
        --dropout)
            DROPOUT=$2
            shift 2
            ;;
        --help)
            echo "Usage: bash run_pipeline.sh [OPTIONS]"
            echo "Options:"
            echo "  --num_folds NUM              Number of folds (default: 5)"
            echo "  --use_important_loss         Use custom loss for important categories"
            echo "  --wandb_project PROJECT      WandB project name"
            echo "  --force_embeddings           Regenerate embeddings"
            echo "  --device DEVICE              GPU device (default: 0)"
            echo "  --batch_size SIZE            Batch size (default: 1024)"
            echo "  --lr RATE                    Learning rate (default: 1e-4)"
            echo "  --epochs N                   Number of epochs (default: 200)"
            echo "  --layer_dims DIMS            Layer dimensions (default: 512 256)"
            echo "  --dropout RATE               Dropout probability (default: 0.4)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "ASTMH Classification Pipeline"
echo "======================================"
echo "Config:"
echo "  Folds: $NUM_FOLDS"
echo "  Important loss: $USE_IMPORTANT_LOSS"
echo "  WandB project: ${WANDB_PROJECT:-none}"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  Layer dims: $LAYER_DIMS"
echo "  Dropout: $DROPOUT"
echo "  Device: $DEVICE"
echo ""

# Step 1: Prepare data
echo "Step 1/5: Preparing data..."
if [ "$FORCE_EMBEDDINGS" = true ]; then
    python 1_prepare_data.py --force
else
    python 1_prepare_data.py
fi
echo "✓ Data prepared"
echo ""

# Step 2: Create splits
echo "Step 2/5: Creating splits..."
python 2_create_splits.py --num_folds $NUM_FOLDS
echo "✓ Splits created"
echo ""

# Step 3: Train models
echo "Step 3/5: Training models on all folds..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "  Training fold $fold..."
    cmd="python 3_train.py --fold $fold --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --layer_dims $LAYER_DIMS --dropout $DROPOUT --devices $DEVICE"
    
    if [ "$USE_IMPORTANT_LOSS" = true ]; then
        cmd="$cmd --use_important_loss"
    fi
    
    if [ ! -z "$WANDB_PROJECT" ]; then
        cmd="$cmd --wandb_project $WANDB_PROJECT --wandb_name fold_${fold}"
    fi
    
    eval $cmd
done
echo "✓ All models trained"
echo ""

# Step 4: Make predictions
echo "Step 4/5: Making predictions..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    echo "  Predicting fold $fold..."
    python 4_predict.py --fold $fold
done
echo "✓ Predictions complete"
echo ""

# Step 5: Evaluate results
echo "Step 5/5: Evaluating results..."
python 5_evaluate.py --num_folds $NUM_FOLDS
echo "✓ Evaluation complete"
echo ""

echo "======================================"
echo "Pipeline finished successfully!"
echo "Results saved to: results/"
echo "======================================"
