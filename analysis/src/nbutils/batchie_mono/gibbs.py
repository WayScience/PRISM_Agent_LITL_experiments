"""Gibbs implementation of the sampler protocol; chain workers never write files."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
from time import perf_counter, process_time

import arviz as az
import numpy as np
from threadpoolctl import threadpool_limits

from .gibbs_kernel import _initialize, _sweep
from .sampling import SamplingConfig, SamplingResult
from .trace import PARAMETER_DIMS, retain_trace


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


def _sample_chain(data, spec, config, seed, blas_threads):
    """Spawn-picklable worker; each chain owns its RNG and mutable state."""
    with threadpool_limits(limits=blas_threads):
        start_cpu = process_time()
        rng = np.random.default_rng(seed)
        state = _initialize(data, spec, rng)
        cell_rows = _group_row_indices(data.model_ids, len(data.models))
        compound_rows = _group_row_indices(data.compound_ids, len(data.compounds))
        samples = {
            name: np.empty((config.raw_draws, *np.shape(value)))
            for name, value in state.items()
        }
        elapsed = np.empty(config.raw_draws)
        for step in range(config.burnin + config.raw_draws):
            start = perf_counter()
            _sweep(
                state,
                data.responses,
                data.model_ids,
                data.compound_ids,
                cell_rows,
                compound_rows,
                spec,
                rng,
            )
            if step >= config.burnin:
                position = step - config.burnin
                for name, value in state.items():
                    samples[name][position] = value
                elapsed[position] = perf_counter() - start
        return samples, elapsed, process_time() - start_cpu


@dataclass(frozen=True)
class GibbsSampler:
    config: SamplingConfig = SamplingConfig()

    def __post_init__(self):
        if type(self.config) is not SamplingConfig:
            raise TypeError(
                "GibbsSampler requires SamplingConfig, without NUTS-only options"
            )

    def sample(self, data, spec) -> SamplingResult:
        config = self.config
        # Preserve the original total BLAS budget and seed derivation.
        workers = min(config.chains, config.cores, config.blas_cores)
        threads = max(1, config.blas_cores // workers)
        seeds = [
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(config.seed).spawn(config.chains)
        ]
        start = perf_counter()
        if workers == 1:
            results = [
                _sample_chain(data, spec, config, seed, threads) for seed in seeds
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [
                    pool.submit(_sample_chain, data, spec, config, seed, threads)
                    for seed in seeds
                ]
                results = [future.result() for future in futures]
        samples = {
            name: np.stack([result[0][name] for result in results])
            for name in results[0][0]
        }
        trace = az.from_dict(
            posterior=samples,
            sample_stats={
                "perf_counter_diff": np.stack([result[1] for result in results])
            },
            observed_data={"response": data.responses},
            coords={
                "cell": data.models,
                "compound": data.compounds,
                "factor": np.arange(spec.rank),
            },
            dims={
                **{name: list(dims) for name, dims in PARAMETER_DIMS.items()},
                "response": ["observation"],
            },
        )
        return SamplingResult(
            trace=retain_trace(
                trace,
                data,
                spec,
                config,
                backend="gibbs",
                parameterization="centered",
                chain_seeds=json.dumps(seeds),
                wall_seconds=perf_counter() - start,
                chain_cpu_seconds=float(sum(result[2] for result in results)),
                chain_workers=workers,
                blas_threads_per_worker=threads,
            )
        )
