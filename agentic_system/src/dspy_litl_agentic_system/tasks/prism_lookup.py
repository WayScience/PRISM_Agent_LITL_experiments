"""
prism_lookup.py

Fast lookups for depmap PRISM data by (drug_id, cell_line).
Backed by a pandas MultiIndex for fast .loc access.
Enables subsetting through pandas .query strings or boolean masks and an
iterator over all entries function to allow building of dataset for
agentic systems.

Classes:
- PrismKey: Immutable (drug, cell_line) pair key.
- PrismLookup: Main class for fast lookups and subsetting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Any, List, Union

import pandas as pd
import numpy as np

@dataclass(frozen=True)
class PrismKey:
    """
    A (drug, cell_line) pair key for depmap PRISM data to quickly
    access IC50 values and other metadata.
    Immutable and hashable.
    """
    drug: str
    cell: str

    def norm(
        self, 
        casefold: bool
    ) -> "PrismKey":
        if casefold:
            return PrismKey(
                self.drug.strip().casefold(), self.cell.strip().casefold())
        return PrismKey(self.drug.strip(), self.cell.strip())

class PrismLookup:
    """
    Fast lookups for depmap PRISM data by (drug_id, cell_line).
    Backed by a pandas MultiIndex for fast .loc access.
    Enables subsetting through pandas .query strings or boolean masks and an
    iterator over all entries function to allow building of dataset for 
    agentic systems.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str = "drug",
        cell_col: str = "cell_line",
        ic50_col: str = "ic50",
        casefold: bool = False,
        validate_unique: bool = True,
    ):
        """
        Initialize the PrismLookup with dataframe and specified column names.
        Default parameters work directly with depmap PRISM secondary screen.

        :param df: drug screening dataframe, e.g. depmap PRISM secondary screen.
        :param drug_col: Column name for drug identifiers. 
            Should uniquely identify drugs in the dataframe.
            Default is "drug" and works with depmap PRISM secondary screen.
        :param cell_col: Column name for cell line identifiers.
            Should uniquely identify cell lines in the dataframe.
            Default is "cell_line" and works with depmap PRISM secondary screen.
        :param ic50_col: Column name for IC50 values.
            Default is "ic50"
        :param casefold: Whether to casefold drug and cell line names
            for case insensitive lookups. Default is False.
        :param validate_unique: Whether to validate that (drug, cell_line)
            pairs are unique in the dataframe. Default is True.
        Raises ValueError if duplicates found.
        """
        self.drug_col = drug_col
        self.cell_col = cell_col
        self.ic50_col = ic50_col
        self.casefold = casefold

        _df = df.copy()
        if casefold:
            _df[drug_col] = _df[drug_col].astype(str).str.strip().str.casefold()
            _df[cell_col] = _df[cell_col].astype(str).str.strip().str.casefold()
        else:
            _df[drug_col] = _df[drug_col].astype(str).str.strip()
            _df[cell_col] = _df[cell_col].astype(str).str.strip()

        # stable sort to ensure consistent ordering across file loads
        self._df = _df.set_index(
            [drug_col, cell_col], verify_integrity=validate_unique).sort_index(
                kind="mergesort")

        if validate_unique and self._df.index.duplicated().any():
            dups = self._df.index[
                self._df.index.duplicated()].unique().tolist()[:5]
            raise ValueError(
                f"Duplicate (drug, cell) keys found; first few: {dups}")

    # -------- main look up methods --------

    def ic50(self, drug: str, cell: str) -> np.float:
        """
        Main lookup function returning the IC50 value for a given
        (drug, cell_line) pair.
        """
        return self._df.loc[(self._norm(drug), self._norm(cell)), self.ic50_col]

    def row(self, drug: str, cell: str) -> pd.Series:
        """
        Return the full row for a given (drug, cell_line) pair.
        Allows for access to other metadata columns.
        """
        return self._df.loc[(self._norm(drug), self._norm(cell))]

    def get(
            self, 
            drug: str, 
            cell: str, 
            default: Optional[Any] = None
        ) -> Optional[Any]:
        """
        Safe get method returning default if (drug, cell_line) pair not found.
        """
        try:
            return self.ic50(drug, cell)
        except KeyError:
            return default

    def get_row(self, drug: str, cell: str) -> Optional[pd.Series]:
        """
        Safe get method returning None if (drug, cell_line) pair not found.
        """
        try:
            return self.row(drug, cell)
        except KeyError:
            return None

    # -------- utility methods --------

    def __contains__(self, key: Union[Tuple[str, str], PrismKey]) -> bool:
        """
        Check if (drug, cell) pair exists in the lookup.
        Supports both tuple and PrismKey inputs.
        Usage: (drug, cell) in lookup or PrismKey(drug, cell) in lookup
        """
        if isinstance(key, PrismKey):
            normalized_key = key.norm(self.casefold)
            return (normalized_key.drug, normalized_key.cell) in self._df.index
        elif isinstance(key, tuple) and len(key) == 2:
            drug, cell = key
            return (self._norm(drug), self._norm(cell)) in self._df.index
        else:
            return False

    def keys(self) -> List[Tuple[str, str]]:
        return self._df.index.to_list()

    def to_frame(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)
    
    # ------- subsetting and iteration --------
    # Useful for building condition specific task queues for agentic systems
    # and lab in the loop history
    def subset(self, query: Union[str, pd.Series]) -> "PrismLookup":
        """
        Return a new PrismLookup object filtered by a pandas query string or 
        boolean mask.
        """
        if isinstance(query, str):
            sub_df = self._df.reset_index().query(query)
        elif isinstance(query, pd.Series):
            # Boolean mask on the reset_index DataFrame
            sub_df = self._df.reset_index()[query]
        elif isinstance(query, pd.DataFrame):
            sub_df = query
        else:
            raise TypeError(
                "query must be a str (pandas query) or boolean mask "
                "(pd.Series) or DataFrame")

        return PrismLookup(
            sub_df,
            drug_col=self.drug_col,
            cell_col=self.cell_col,
            ic50_col=self.ic50_col,
            casefold=self.casefold,
            validate_unique=True,
        )

    def __iter__(self):
        """
        Iterate over all entries.
        Yields (drug, cell, ic50, row: pd.Series).
        """
        for (drug, cell), row in self._df.iterrows():
            yield (drug, cell, row[self.ic50_col], row)

    def _norm(self, s: str) -> str:
        return s.strip().casefold() if self.casefold else s.strip()
