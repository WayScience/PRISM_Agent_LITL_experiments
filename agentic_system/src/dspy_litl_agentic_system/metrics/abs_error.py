"""
abs_error.py

Absolute error metrics for IC50 predictions.

Absolute error is the absolute difference between predicted and true values,
providing a direct measure of prediction accuracy in the original units.
"""

import numpy as np
from typing import Union, List


def absolute_error(
    y_true: Union[float, List[float], np.ndarray],
    y_pred: Union[float, List[float], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate absolute error between predicted and true IC50 values.
    
    Absolute error = |y_pred - y_true|
    
    :param y_true: True IC50 values (can be a single value or an array)
    :param y_pred: Predicted IC50 values (can be a single value or an array)
    :return: Absolute error(s). Single float if inputs are scalars, 
        array if inputs are arrays.
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Input shapes must match: y_true {y_true.shape} "
            f"vs y_pred {y_pred.shape}"
        )
    
    # Calculate absolute error
    abs_err = np.abs(y_pred - y_true)
    
    # Return scalar if input was scalar
    if abs_err.ndim == 0:
        return float(abs_err)
    
    return abs_err
