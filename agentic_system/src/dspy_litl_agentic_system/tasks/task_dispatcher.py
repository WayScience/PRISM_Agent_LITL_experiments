"""
task_dispatcher.py

Implements a dispatch queue for (drug, cell) tasks from a PrismLookup.
Useful for orchestrating agentic systems that process drug-cell pairs
in a controlled order, with optional shuffling and progress tracking, 
so we can observe the effects of task order on outcomes and how
an agent may improve over task iterations given feedback and calibration.

Classes:
- DispatchItem: Immutable data class representing a single dispatch item.
- PrismDispatchQueue: Main class for managing the dispatch queue.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import random

import pandas as pd

from .prism_lookup import PrismLookup

@dataclass(frozen=True)
class DispatchItem:
    drug: str
    cell: str
    ic50: Any
    row: pd.Series # full row reference

class PrismDispatchQueue:
    """
    Iterates/dispatches (drug, cell) tasks from a PrismLookup in a 
        controlled order.
    - Accepts an immutable PrismLookup
    - Builds an internal ordered list of keys (no data copy)
    - Optional seeded shuffle
    - Dispatches one item at a time and tracks progress
    - Only operates on the index list to determine order and track progress,
        uses PrismLookup backend for actual data retrieval
    """

    def __init__(
        self,
        lookup: PrismLookup,
        order: Optional[Iterable[Tuple[str, str]]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Create a dispatch queue from a PrismLookup.

        :param lookup: PrismLookup instance (immutable)
        :param order: Optional iterable of (drug, cell) tuples to define a 
            custom order. If None, uses all keys in the lookup in its 
            canonical order.
        :param shuffle: Whether to shuffle the order 
            (after applying custom order if given).
        :param seed: Optional random seed for deterministic shuffling.
        """

        self.lookup: PrismLookup = lookup
        # materialize keys from lookup to lock an initial ordering
        # later to be optionally shuffled
        keys = list(order) if order is not None else list(self.lookup.keys())

        # drop unknown keys if a custom order was provided
        if order is not None:
            unknown = [
                (d, c) for (d, c) in keys if not (d, c) in self.lookup]
            if unknown:
                error_msg = f"These (drug, cell) keys are not in the lookup: {unknown[:5]}"
                if len(unknown) > 5:
                    error_msg += f" ... ({len(unknown)} total)"
                raise KeyError(error_msg)

        # seeded shuffle for reproducibility
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(keys)

        # attributes for tracking state
        self._keys: List[Tuple[str, str]] = keys
        self._cursor: int = 0
        self._seed: Optional[int] = seed
        self._shuffled: bool = shuffle

    # -------- core dispatch API --------
    # called by the agentic system orchestrator to get next task
    # as the return will contain the truth the agent itself
    # should not be allowed to access these methods
    def has_next(self) -> bool:
        return self._cursor < len(self._keys)

    def peek(self) -> Optional[DispatchItem]:
        """Look at the next item without advancing."""
        if not self.has_next():
            return None
        d, c = self._keys[self._cursor]
        row = self.lookup.row(d, c)
        return DispatchItem(
            drug=d, cell=c, ic50=row[self.lookup.ic50_col], row=row)

    def dispatch(self) -> Optional[DispatchItem]:
        """
        Return the next (drug, cell, ic50, row) and advance the cursor.
        """
        if not self.has_next():
            return None
        d, c = self._keys[self._cursor]
        self._cursor += 1
        row = self.lookup.row(d, c)  # always pull from backend
        return DispatchItem(
            drug=d, cell=c, ic50=row[self.lookup.ic50_col], row=row)

    # -------- progress tracker --------

    @property
    def index(self) -> int:
        """
        0-based index of next item to dispatch.
        """
        return self._cursor

    @property
    def remaining(self) -> int:
        return len(self._keys) - self._cursor

    @property
    def total(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> List[Tuple[str, str]]:
        """
        Return the frozen order of keys for this queue.
        """
        return list(self._keys)
    
    # -------- control --------
    # only reset with shuffling is supported and no rewinding/skipping
    # as we don't anticipate that being useful in agentic system experimetnation
    # where we will only replicate full runs on the same agentic system with 
    # different orders of task queues and same task queue for different agentic 
    # systems
    def reset(
        self, 
        shuffle: Optional[bool] = None, 
        seed: Optional[int] = None
    ) -> None:
        """
        Reset cursor; optionally reshuffle.
        
        :param shuffle: Whether to reshuffle. If None, retains current
            shuffle state.
        :param seed: Optional new seed for reshuffling. If None, retains
            current seed.
        """
        if shuffle is None:
            shuffle = self._shuffled
        self._cursor = 0
        if shuffle:
            rng = random.Random(seed if seed is not None else self._seed)
            rng.shuffle(self._keys)
            self._shuffled = True
            if seed is not None:
                self._seed = seed
        else:
            self._shuffled = False

    # -------- recreation from configs --------

    def to_state(self) -> Dict[str, Any]:
        """Serialize the dispatch state (order, cursor, seed, results)."""
        return {
            "keys": list(self._keys),
            "cursor": self._cursor,
            "seed": self._seed,
            "shuffled": self._shuffled,
        }

    @classmethod
    def from_state(cls, lookup, state: Dict[str, Any]) -> "PrismDispatchQueue":
        """
        Recreate a queue from a saved state.
        NOTE: Will validate that all keys exist in the provided lookup.
        """
        q = cls(
            lookup, order=state["keys"], shuffle=False, seed=state.get("seed"))
        q._cursor = int(state.get("cursor", 0))
        q._shuffled = bool(state.get("shuffled", False))
        q._seed = state.get("seed", None)
        # validate cursor bounds
        if not (0 <= q._cursor <= len(q._keys)):
            raise ValueError(
                f"Invalid cursor {q._cursor} for {len(q._keys)} items.")
        return q
