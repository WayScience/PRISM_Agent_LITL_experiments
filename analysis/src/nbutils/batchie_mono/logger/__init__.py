"""Public API for experiment logging and input fingerprinting."""


from nbutils.batchie_mono.logger.logger import ExperimentLogger
from nbutils.batchie_mono.logger.artifacts import Artifact, ArtifactStore

__all__ = [
    "ExperimentLogger",
    "Artifact",
    "ArtifactStore",
    "LoggedModel",
    "fingerprint_responses",
]


def __getattr__(name):
    if name == "fingerprint_responses":
        from nbutils.batchie_mono.logger.fingerprints import fingerprint_responses

        return fingerprint_responses
    # Pandas is needed only for the optional simulation adapter.
    if name == "LoggedModel":
        from nbutils.batchie_mono.logger.integration import LoggedModel

        return LoggedModel
    raise AttributeError(name)
