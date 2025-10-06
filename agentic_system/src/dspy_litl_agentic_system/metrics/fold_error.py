"""
fold_error.py

Fold error metric for IC50 predictions.

Fold error is defined as the maximum of (pred/true, true/pred), which represents
how many times larger one value is compared to the other. A fold error of 1 
indicates perfect prediction, while larger values indicate worse performance.
"""

import numpy as np
from typing import Union, List


def fold_error(
    y_true: Union[float, List[float], np.ndarray], 
    y_pred: Union[float, List[float], np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate fold error between predicted and true IC50 values.
    
    Fold error = max(y_pred/y_true, y_true/y_pred)
    
    :param y_true: True IC50 values (can be a single value or an array)
    :param y_pred: Predicted IC50 values (can be a single value or an array)
    :param epsilon: Small value to avoid division by zero
    :return: Fold error(s). Single float if inputs are scalars, 
        array if inputs are arrays.
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input shapes must match: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("IC50 values must be non-negative")
    
    # Add epsilon to avoid division by zero
    y_true_safe = y_true + epsilon
    y_pred_safe = y_pred + epsilon
    
    # Calculate fold error
    ratio1 = y_pred_safe / y_true_safe
    ratio2 = y_true_safe / y_pred_safe
    
    fold_err = np.maximum(ratio1, ratio2)
    
    # Return scalar if input was scalar
    if fold_err.ndim == 0:
        return float(fold_err)
    
    return fold_err
