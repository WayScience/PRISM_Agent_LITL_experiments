"""
Small numerical utilities sharable across response-model implementations.
"""

import numpy as np


def _group_row_indices(ids: np.ndarray, group_count: int) -> list[np.ndarray]:
    """
    Group observation positions by nonnegative integer ID.
    IDs are assumed to lie in [0, group_count).

    :param ids: Array of nonnegative integer IDs for each observation.
    :param group_count: Total number of groups.
    :return: List of arrays, each containing the row indices for a group.
    """
    order = np.argsort(ids, kind="stable")
    counts = np.bincount(ids, minlength=group_count)
    return list(np.split(order, np.cumsum(counts)[:-1]))


def _ridge_solve(
    design: np.ndarray, 
    target: np.ndarray, 
    penalty: float
) -> np.ndarray:
    """
    Solve penalized normal equations for a bias + factor design.

    The first column is assumed to be the intercept and receives 0.1 *
    penalty; all other coefficients receive raw penalty. 

    :param design: Design matrix of shape (observations, coefficients).
    :param target: Target vector of shape (observations,).
    :param penalty: Ridge regularization penalty.
    :return: Solution vector of shape (coefficients,).
    """
    gram = design.T @ design
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = penalty * 0.1
    return np.linalg.solve(gram + regularizer, design.T @ target)


def _factor_mean(
    global_offset: float,
    cell_offsets: np.ndarray,
    compound_offsets: np.ndarray,
    cell_embeddings: np.ndarray,
    compound_embeddings: np.ndarray,
    model_ids: np.ndarray,
    compound_ids: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the fixed-dose mean at aligned observed index pairs.
    Offsets and embeddings are model parameters, 
        whereas model_ids and compound_ids index the (paired) observations.

    :param global_offset: Scalar global offset.
    :param cell_offsets: Array of cell-specific offsets.
    :param compound_offsets: Array of compound-specific offsets.
    :param cell_embeddings: Array of cell embeddings.
    :param compound_embeddings: Array of compound embeddings.
    :param model_ids: Array of row indices for the observations.
    :param compound_ids: Array of column indices for the observations.
    :return: Array of evaluated means for each observation.
    """
    return (
        global_offset
        + cell_offsets[model_ids]
        + compound_offsets[compound_ids]
        + np.einsum(
            "nr,nr->n",
            cell_embeddings[model_ids],
            compound_embeddings[compound_ids],
        )
    )


def _sample_from_precision(
    precision: np.ndarray,
    mean_part: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a Gaussian vector (n-dimensional sample) from specified multi-var distribution.
    
    `precision` refers to the precision matrix (inverse of covariance matrix).
    `mean_part` refers to the unnormalized mean vector.
    Together these two define completely a multivariate Gaussian distribution.
    
    For positive-definite precision P and vector h = ``mean_part``, sample
    from N(P^-1 h, P^-1). If P = L L^T, solving L^T x = z for standard normal
    z gives the correct covariance without explicitly forming P^-1.

    :param precision: Precision matrix describing the Gaussian distribution.
    :param mean_part: Unnormalized mean vector describing the Gaussian distribution.
    :param rng: Random number generator.
    :return: Sampled Gaussian vector.
    """
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(precision, mean_part)
    return mean + np.linalg.solve(chol.T, rng.normal(size=len(mean_part)))
