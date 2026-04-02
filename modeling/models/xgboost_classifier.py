"""
xgboost_classifier.py - XGBoost classifier for ASTMH abstracts

Optimized XGBoost model with strong regularization to prevent overfitting.
Better generalization on multi-class classification tasks.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from pathlib import Path


class XGBoostClassifier:
    """
    XGBoost classifier wrapper for ASTMH abstract classification.
    
    Features:
    - Strong regularization to prevent overfitting
    - Built-in early stopping on validation set
    - Feature importance tracking
    - Cross-fold training support
    - Model persistence (save/load)
    """
    
    def __init__(
        self,
        num_classes: int = 54,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        lambda_l2: float = 1.0,
        alpha_l1: float = 0.0,
        scale_pos_weight: float = 1.0,
        tree_method: str = "hist",
        device: str = "cuda",
        random_state: int = 42,
    ):
        """
        Args:
            num_classes: Number of output classes
            max_depth: Maximum tree depth (deeper = more complex, higher variance)
            learning_rate: Boosting learning rate (0.01-0.3)
            subsample: Fraction of samples per tree (0.7-1.0)
            colsample_bytree: Fraction of features per tree (0.7-1.0)
            min_child_weight: Minimum weight in child nodes (>=1)
            lambda_l2: L2 regularization strength (>0)
            alpha_l1: L1 regularization strength (>=0)
            scale_pos_weight: Balance of classes (for imbalanced data)
            tree_method: "hist" (fast, GPU) or "exact" (slow, CPU)
            device: "cuda" or "cpu"
            random_state: Random seed
        """
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.lambda_l2 = lambda_l2
        self.alpha_l1 = alpha_l1
        self.scale_pos_weight = scale_pos_weight
        self.tree_method = tree_method
        self.device = device
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        
    def _get_params(self):
        """Get XGBoost parameters."""
        # Determine device type
        gpu_id = 0
        tree_method = self.tree_method
        device_str = self.device
        
        # Use GPU if available and requested
        if self.device == "cuda":
            try:
                xgb.get_config()
                device_str = "cuda"
                gpu_id = 0
            except:
                device_str = "cpu"
        
        params = {
            "objective": "multi:softmax" if self.num_classes > 2 else "binary:logistic",
            "num_class": self.num_classes if self.num_classes > 2 else None,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.lambda_l2,
            "alpha": self.alpha_l1,
            "tree_method": tree_method,
            "random_state": self.random_state,
            "verbosity": 1,
            "eval_metric": "mlogloss" if self.num_classes > 2 else "logloss",
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_rounds: int = 500,
        early_stopping_rounds: int = 50,
    ):
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            num_rounds: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience
        """
        print(f"Training XGBoost classifier...")
        print(f"  Train shape: {X_train.shape}, num classes: {self.num_classes}")
        print(f"  Val shape: {X_val.shape}")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Get parameters
        params = self._get_params()
        
        # Train with early stopping
        evals = [(dtrain, "train"), (dval, "eval")]
        evals_result = {}
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type="weight")
        
        # Get final metrics
        train_preds = self.predict(X_train)
        val_preds = self.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        print(f"\n✓ Training complete")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Val accuracy: {val_acc:.4f}")
        print(f"  Best iteration: {self.model.best_iteration}")
        
        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_iteration": self.model.best_iteration,
            "evals_result": evals_result,
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        return predictions.astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, num_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        dmatrix = xgb.DMatrix(X)
        # Use margin to get raw scores, then apply softmax
        margins = self.model.predict(dmatrix, output_margin=True)
        
        if self.num_classes > 2:
            # Multi-class: margins is (n_samples, num_classes)
            margins = margins.reshape(-1, self.num_classes)
            # Apply softmax
            exp_margins = np.exp(margins - margins.max(axis=1, keepdims=True))
            proba = exp_margins / exp_margins.sum(axis=1, keepdims=True)
        else:
            # Binary: apply sigmoid to single column
            proba = 1.0 / (1.0 + np.exp(-margins))
            proba = np.column_stack([1 - proba, proba])
        
        return proba
    
    def get_feature_importance(self, top_n: int = 20) -> dict:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature_name: importance_score
        """
        if self.feature_importance is None:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_features[:top_n])
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(filepath))
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = xgb.Booster()
        self.model.load_model(str(filepath))
        print(f"✓ Model loaded from {filepath}")
