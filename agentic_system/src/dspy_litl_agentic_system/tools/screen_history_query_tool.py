from __future__ import annotations
from typing import List, Tuple, Optional, Any, Union
import pandas as pd

from ..tasks.prism_lookup import PrismLookup


class ScreenHistoryQueryTool:
    """
    A restricted PrismLookup that serves as a tool for agents to query
    screening histories. Only contains drug-cell combinations that have
    been explicitly revealed to the agent.
    
    Acts like a PrismLookup but with controlled data access.
    """
    
    def __init__(
        self,
        drug_col: str = "drug",
        cell_col: str = "cell_line", 
        ic50_col: str = "ic50",
        casefold: bool = False
    ):
        """Initialize an empty screening history tool."""
        self.drug_col = drug_col
        self.cell_col = cell_col
        self.ic50_col = ic50_col
        self.casefold = casefold
        
        # Start with empty DataFrame with proper structure
        empty_df = pd.DataFrame(columns=[drug_col, cell_col, ic50_col])
        self._lookup = PrismLookup(
            empty_df,
            drug_col=drug_col,
            cell_col=cell_col, 
            ic50_col=ic50_col,
            casefold=casefold,
            validate_unique=True
        )
    
    def add_from_lookup(self, lookup: PrismLookup) -> None:
        """Add all entries from a PrismLookup (e.g., a subset)."""
        if len(lookup) == 0:
            return
            
        # Get the DataFrame and combine with existing data
        new_df = lookup.to_frame().reset_index()
        if len(self._lookup) > 0:
            existing_df = self._lookup.to_frame().reset_index()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=[self.drug_col, self.cell_col], keep='last')
        else:
            combined_df = new_df
            
        # Rebuild the lookup
        self._lookup = PrismLookup(
            combined_df,
            drug_col=self.drug_col,
            cell_col=self.cell_col,
            ic50_col=self.ic50_col,
            casefold=self.casefold,
            validate_unique=True
        )
    
    def add_item(self, drug: str, cell: str, ic50: Any, row: pd.Series) -> None:
        """Add a single drug-cell combination from a dispatch item."""
        # Create a single-row DataFrame
        item_data = row.to_dict()
        item_data[self.drug_col] = drug
        item_data[self.cell_col] = cell
        item_data[self.ic50_col] = ic50
        
        item_df = pd.DataFrame([item_data])
        temp_lookup = PrismLookup(
            item_df,
            drug_col=self.drug_col,
            cell_col=self.cell_col,
            ic50_col=self.ic50_col, 
            casefold=self.casefold,
            validate_unique=True
        )
        
        self.add_from_lookup(temp_lookup)
    
    def add_dispatch_item(self, dispatch_item) -> None:
        """Add a DispatchItem directly to the screening history."""
        self.add_item(
            dispatch_item.drug, 
            dispatch_item.cell, 
            dispatch_item.ic50, 
            dispatch_item.row
        )
    
    def add_from_dispatch_queue(self, queue, keys: List[Tuple[str, str]]) -> None:
        """Add specific items from a dispatch queue by their keys."""
        for drug, cell in keys:
            if queue.lookup.contains(drug, cell):
                row = queue.lookup.row(drug, cell)
                ic50 = row[queue.lookup.ic50_col]
                self.add_item(drug, cell, ic50, row)
    
    # Agent-facing query interface (same as PrismLookup)
    def ic50(self, drug: str, cell: str) -> Any:
        """Get IC50 value for a drug-cell combination."""
        return self._lookup.ic50(drug, cell)
    
    def get(self, drug: str, cell: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get IC50 value with default fallback."""
        return self._lookup.get(drug, cell, default)
    
    def row(self, drug: str, cell: str) -> pd.Series:
        """Get full row for a drug-cell combination.""" 
        return self._lookup.row(drug, cell)
    
    def get_row(self, drug: str, cell: str) -> Optional[pd.Series]:
        """Get full row with None fallback."""
        return self._lookup.get_row(drug, cell)
    
    def contains(self, drug: str, cell: str) -> bool:
        """Check if drug-cell combination is in the revealed history."""
        return self._lookup.contains(drug, cell)
    
    def keys(self) -> List[Tuple[str, str]]:
        """Get all revealed drug-cell combinations."""
        return self._lookup.keys()
    
    def to_frame(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._lookup.to_frame()
    
    def __len__(self) -> int:
        """Number of revealed drug-cell combinations."""
        return len(self._lookup)
    
    def subset(self, query: Union[str, pd.Series]) -> PrismLookup:
        """Create a subset of the revealed history."""
        return self._lookup.subset(query)
    
    def __iter__(self):
        """Iterate over revealed entries."""
        return iter(self._lookup)
