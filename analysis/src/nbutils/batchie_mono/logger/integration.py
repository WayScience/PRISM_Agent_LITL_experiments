"""Record model calls without importing Batchie models or sampler backends.

``LoggedModel`` adapts a caller-owned fit/predict model to ``ExperimentLogger``.
It records actual successful calls only; acquisition and truth revelation remain
the controller's responsibility. Writes commit independently, so a partially
recorded trajectory is not proof of completion or an automatic recovery point.
"""

from uuid import uuid4


class LoggedModel:
    """Log fit/predict calls and forward other attributes to a wrapped model.

    One run identifies one fit; a trajectory groups this wrapper's adaptive
    fits. The model exposes ``inference_data_`` after fit and ``prediction_seed``.
    Fit and predict through this wrapper, not directly through the model, to
    keep recorded state aligned. Other forwarded methods are not logged.

    Errors propagate without retries or failed-fit records. The wrapper owns
    neither the model nor the logger and does not close either. Attribute reads
    are forwarded, but assignments on the wrapper are not: change model settings
    on ``model`` itself. This mutable wrapper is intended for one controller.
    """

    def __init__(
        self,
        model,
        logger,
        experiment_name,
        *,
        trajectory_id=None,
        first_batch=0,
        metadata=None,
    ):
        """Configure call recording without fitting or writing to the database.

        Args:
            model: Model with fit/predict methods and the attributes above.
            logger: ExperimentLogger-compatible persistence object.
            experiment_name: Nonblank string grouping related trajectories.
            trajectory_id: Optional grouping label; None generates a new UUID.
            first_batch: Nonnegative Python integer starting the fit counter.
            metadata: Optional mapping shallow-copied into fit metadata.

        Raises:
            ValueError: Invalid experiment name or first batch index.
        """
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a nonempty string")
        if isinstance(first_batch, bool) or not isinstance(first_batch, int) or first_batch < 0:
            raise ValueError("first_batch must be a nonnegative integer")
        self.model, self.logger = model, logger
        self.experiment_name = experiment_name
        self.trajectory_id = str(uuid4()) if trajectory_id is None else trajectory_id
        self.next_batch = first_batch
        self.metadata = dict(metadata or {})
        self.last_run_id = self.last_prediction_id = None

    def __getattr__(self, name):
        """Forward missing attribute reads and methods, without logging calls.

        AttributeError propagates if the model lacks the requested name. Direct
        base lookup avoids recursive forwarding if ``model`` is not yet set.
        """
        return getattr(object.__getattribute__(self, "model"), name)

    def fit(self, observed):
        """Fit once, persist the retained trace, and return this wrapper.

        Args:
            observed: Training input forwarded unchanged to ``model.fit``.

        Both last IDs are cleared before fitting. Only a successful log advances
        ``next_batch`` and sets ``last_run_id``. Fit metadata includes the model's
        current prediction seed, overriding a caller field of the same name.
        Model/serialization/database errors propagate; the model may have fitted
        even if logging fails, but prediction remains disabled in this wrapper.
        Earlier successful records are not deleted or rolled back.
        """
        self.last_run_id = self.last_prediction_id = None
        self.model.fit(observed)
        self.last_run_id = self.logger.log_run(
            self.experiment_name,
            self.model.inference_data_,
            trajectory_id=self.trajectory_id,
            batch_index=self.next_batch,
            metadata={**self.metadata, "prediction_seed": self.model.prediction_seed},
        )
        self.next_batch += 1
        return self

    def predict(self, candidates):
        """Predict once, persist the output frame, and return it unchanged.

        Args:
            candidates: Input forwarded unchanged to ``model.predict``.

        Raises:
            RuntimeError: No fit was successfully recorded through this wrapper.

        Clears ``last_prediction_id`` before attempting the call. Prediction or
        logging failures propagate, preserving the last fit ID but not a stale
        prediction ID. Each successful call records the model's current seed.
        """
        self.last_prediction_id = None
        if self.last_run_id is None:
            raise RuntimeError("Call fit through LoggedModel before predict")
        prediction = self.model.predict(candidates)
        self.last_prediction_id = self.logger.log_prediction(
            self.last_run_id, prediction,
            metadata={"prediction_seed": self.model.prediction_seed},
        )
        return prediction
