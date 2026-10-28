"""Small sampler contract: encoded observations + model specification -> trace."""

from __future__ import annotations

from typing import Annotated, TYPE_CHECKING, Literal, Protocol

import arviz as az
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

from .spec import ModelSpec, PositiveInt

if TYPE_CHECKING:
    from .data import EncodedObservations

NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
AcceptanceProbability = Annotated[FiniteFloat, Field(gt=0, lt=1)]


class SamplingConfig(BaseModel):
    """Draw counts are per chain; burnin is discarded, then [::thin] is kept."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", validate_default=True)

    burnin: NonnegativeInt = 40
    posterior_samples: PositiveInt = 30
    thin: PositiveInt = 2
    chains: PositiveInt = 1
    seed: NonnegativeInt = 0
    cores: PositiveInt = 1
    blas_cores: PositiveInt = 1

    @property
    def raw_draws(self):
        return self.posterior_samples * self.thin


class NUTSConfig(SamplingConfig):
    target_accept: AcceptanceProbability = 0.9
    max_treedepth: PositiveInt = 10
    parameterization: Literal["centered", "noncentered"] = "centered"


class SamplingResult(BaseModel):
    """Authoritative retained trace; preserve chain/draw axes for later analysis.

    No database handles, fitted model, predictions, or flattened copies cross
    this boundary. Backend-native statistics remain in trace.sample_stats.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    trace: az.InferenceData


class PosteriorSampler(Protocol):
    def sample(self, data: EncodedObservations, spec: ModelSpec) -> SamplingResult:
        """Sample the specified posterior without mutating data or specification."""
        ...
