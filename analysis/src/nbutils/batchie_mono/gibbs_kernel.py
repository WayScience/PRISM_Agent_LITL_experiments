"""
Original-coordinate blocked Gibbs kernel; no orchestration or persistence.
"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


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
    if not np.isfinite(precision).all() or not np.isfinite(mean_part).all():
        raise ValueError("Gaussian conditional parameters must be finite")
    if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Gaussian precision must be symmetric")
    chol = np.linalg.cholesky(precision)
    mean = cho_solve((chol, True), mean_part)
    draw = mean + solve_triangular(chol.T, rng.normal(size=len(mean_part)), lower=False)
    if not np.isfinite(draw).all():
        raise FloatingPointError("Nonfinite Gaussian conditional draw")
    return draw


def _gaussian_conditional(design, residual, observation_precision,
                          prior_precision, prior_linear):
    """Return P, h for Gaussian regression; prior terms may be dense/nonzero."""
    if not np.isfinite(observation_precision) or observation_precision <= 0:
        raise ValueError("Observation precision must be finite and positive")
    return (prior_precision + observation_precision * (design.T @ design),
            prior_linear + observation_precision * (design.T @ residual))


def _gamma_conditional(prior_shape, prior_rate, count, sum_squares):
    """Gaussian precision full conditional, explicitly in shape-rate form."""
    if (not np.isfinite(prior_shape) or prior_shape <= 0
            or not np.isfinite(prior_rate) or prior_rate <= 0
            or not np.isfinite(count).all() or np.any(np.asarray(count) < 0)
            or not np.isfinite(sum_squares).all() or np.any(np.asarray(sum_squares) < 0)):
        raise ValueError("Invalid Gamma conditional statistics")
    return prior_shape + np.asarray(count) / 2, prior_rate + np.asarray(sum_squares) / 2


def _sample_gamma(shape, rate, rng):
    """Draw Gamma(shape, rate) using NumPy's scale convention, without clipping."""
    if (not np.isfinite(shape).all() or np.any(np.asarray(shape) <= 0)
            or not np.isfinite(rate).all() or np.any(np.asarray(rate) <= 0)):
        raise ValueError("Gamma shape and rate must be finite and positive")
    draw = rng.gamma(shape, scale=1.0 / np.asarray(rate))
    if not np.isfinite(draw).all() or np.any(draw <= 0):
        raise FloatingPointError("Gamma draw must be finite and positive")
    return draw


def _initialize(data, spec, rng):
    """Independent joint prior draw; shared compound precision includes offset."""
    k, j, r = len(data.models), len(data.compounds), spec.rank
    s, t = spec.prior_shape, spec.prior_rate
    state = {
        "cell_offset_precision": _sample_gamma(s, t, rng),
        "dimension_precisions": _sample_gamma(np.full(r, s), t, rng),
        "compound_precisions": _sample_gamma(np.full(j, s), t, rng),
        "observation_precision": _sample_gamma(s, t, rng),
        "global_offset": rng.normal(scale=spec.global_precision**-0.5),
    }
    state["cell_offsets"] = rng.normal(size=k) / np.sqrt(state["cell_offset_precision"])
    state["compound_offsets"] = rng.normal(size=j) / np.sqrt(
        state["compound_precisions"]
    )
    state["cell_embeddings"] = rng.normal(size=(k, r)) / np.sqrt(
        state["dimension_precisions"]
    )
    state["compound_embeddings"] = rng.normal(size=(j, r)) / np.sqrt(
        state["compound_precisions"][:, None]
    )
    return state


def _mean(state, model_ids, compound_ids):
    return _factor_mean(
        *(
            state[name]
            for name in (
                "global_offset",
                "cell_offsets",
                "compound_offsets",
                "cell_embeddings",
                "compound_embeddings",
            )
        ),
        model_ids,
        compound_ids,
    )


def _sweep(
    state, responses, model_ids, compound_ids, cell_rows, compound_rows, spec, rng
):
    """Intercept, joint cell blocks, joint compound blocks, all four precisions."""
    a, b = state["cell_offsets"], state["compound_offsets"]
    u, v = state["cell_embeddings"], state["compound_embeddings"]
    tau = state["observation_precision"]
    residual = (
        responses - _mean(state, model_ids, compound_ids) + state["global_offset"]
    )
    p = np.array([[spec.global_precision + len(responses) * tau]])
    h = np.array([tau * residual.sum()])
    state["global_offset"] = _sample_from_precision(p, h, rng)[0]
    alpha = state["global_offset"]
    zero = np.zeros(spec.rank + 1)
    cell_prior = np.diag(
        np.r_[state["cell_offset_precision"], state["dimension_precisions"]]
    )
    identity = np.eye(spec.rank + 1)
    for i, rows in enumerate(cell_rows):
        drugs = compound_ids[rows]
        design = np.column_stack((np.ones(len(rows)), v[drugs]))
        p, h = _gaussian_conditional(
            design, responses[rows] - alpha - b[drugs], tau, cell_prior, zero
        )
        block = _sample_from_precision(p, h, rng)
        a[i], u[i] = block[0], block[1:]
    for j, rows in enumerate(compound_rows):
        cells = model_ids[rows]
        design = np.column_stack((np.ones(len(rows)), u[cells]))
        p, h = _gaussian_conditional(
            design,
            responses[rows] - alpha - a[cells],
            tau,
            state["compound_precisions"][j] * identity,
            zero,
        )
        block = _sample_from_precision(p, h, rng)
        b[j], v[j] = block[0], block[1:]
    residual = responses - _mean(state, model_ids, compound_ids)
    statistics = (
        ("cell_offset_precision", len(a), a @ a),
        ("dimension_precisions", len(a), np.square(u).sum(axis=0)),
        ("compound_precisions", spec.rank + 1, b**2 + np.square(v).sum(axis=1)),
        ("observation_precision", len(responses), residual @ residual),
    )
    for name, count, squares in statistics:
        shape, rate = _gamma_conditional(
            spec.prior_shape, spec.prior_rate, count, squares
        )
        state[name] = _sample_gamma(shape, rate, rng)
