"""
Portable immutable artifacts resulting from completed actions.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from hashlib import sha256
from tempfile import NamedTemporaryFile

from nbutils.batchie_mono.logger._storage import uid


@dataclass(frozen=True)
class Artifact:
    """
    A finalized file ready to attach to a completed action (model fitting, prediction making etc.)

    The writer receives a temporary Path and must close it before returning. 
    Metadata must be JSON-compatible and should describe the serializer/version.
    """

    relative_path: str
    format: str
    sha256: str
    size_bytes: int
    role: str
    metadata: dict = field(default_factory=dict)


class ArtifactStore:
    """
    Manages the storage and retrieval of Artifact instances on the filesystem.
    Ensures that artifacts are written atomically and can be verified for integrity.
    """
    def __init__(self, database_path: str | Path):
        """Initialize the artifact store with a given database path."""
        self.base = Path(database_path).resolve().parent
        # Namespace prevents different DBs in one directory sharing artifact files.
        self.root = self.base / "artifacts" / Path(database_path).name
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, role: str, format: str, writer: callable, *, metadata=None):
        """Write an artifact atomically to the store"""
        if not format.isalnum():
            raise ValueError("format must be an alphanumeric file extension")
        with NamedTemporaryFile(dir=self.root, suffix="." + format, delete=False) as f:
            temp = Path(f.name)
        try:
            writer(temp)
            with temp.open("rb") as f:
                os.fsync(f.fileno())
                checksum = sha256()
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    checksum.update(block)
            final = self.root / (uid() + "." + format)
            temp.replace(final)
            # Persist the directory entry as well on POSIX.
            if os.name == "posix":
                fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return Artifact(
                str(final.relative_to(self.base)),
                format,
                checksum.hexdigest(),
                final.stat().st_size,
                role,
                metadata or {},
            )
        finally:
            temp.unlink(missing_ok=True)

    def verify(self, artifact: Artifact):
        """Resolve a relative reference and verify bytes before loading."""
        path = (self.base / artifact.relative_path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path escapes this database artifact directory")
        checksum = sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                checksum.update(block)
        if (
            path.stat().st_size != artifact.size_bytes
            or checksum.hexdigest() != artifact.sha256
        ):
            raise ValueError(f"Artifact checksum mismatch: {artifact.relative_path}")
        return path


def write_inference_data(store: ArtifactStore, trace):
    """Retain all supplied groups/draws; never thin or modify the caller's trace."""
    if not hasattr(trace, "posterior") or any(
        trace.posterior.sizes.get(x, 0) < 1 for x in ("chain", "draw")
    ):
        raise ValueError("Expected a nonempty posterior with chain/draw dimensions")
    return store.write(
        "posterior",
        "nc",
        lambda path: trace.to_netcdf(str(path), engine="h5netcdf"),
        metadata={"serializer": "arviz.InferenceData", "version": 1},
    )


def load_inference_data(store: ArtifactStore, artifact: Artifact):
    """Optional dependency is imported only when loading a Bayesian artifact."""
    import arviz as az

    trace = az.from_netcdf(str(store.verify(artifact)), engine="h5netcdf")
    try:
        for group in trace.groups():
            trace[group].load()
    finally:
        trace.close()
    return trace
