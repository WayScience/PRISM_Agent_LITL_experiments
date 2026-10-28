from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

PositiveInt = Annotated[StrictInt, Field(ge=1)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]


class ModelSpec(BaseModel):
    """Gaussian factor model with Gamma(shape, rate) precision priors."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", validate_default=True)

    rank: PositiveInt = 4
    prior_shape: PositiveFloat = 1.1
    prior_rate: PositiveFloat = 1.1
    global_precision: PositiveFloat = 0.01
