import pytest
import pandas as pd
from typing import List, Tuple, Any, Optional
from unittest.mock import Mock

from dspy_litl_agentic_system.tasks.task_dispatcher import PrismDispatchQueue, DispatchItem


class FakePrismLookup:
    """Mock PrismLookup for testing dispatcher."""
    
    def __init__(self, data: List[Tuple[str, str, float]], ic50_col: str = "ic50"):
        self.ic50_col = ic50_col
        self._data = {}
        self._keys = []
        
        for drug, cell, ic50 in data:
            key = (drug, cell)
            self._keys.append(key)
            # Create a mock Series with ic50 and some additional columns
            row_data = {
                ic50_col: ic50,
                "drug_name": drug,
                "cell_line": cell,
                "other_col": f"data_{drug}_{cell}"
            }
            self._data[key] = pd.Series(row_data)
    
    def keys(self) -> List[Tuple[str, str]]:
        return list(self._keys)
    
    def contains(self, drug: str, cell: str) -> bool:
        return (drug, cell) in self._data
    
    def row(self, drug: str, cell: str) -> pd.Series:
        if (drug, cell) not in self._data:
            raise KeyError(f"Key ({drug}, {cell}) not found")
        return self._data[(drug, cell)]


class TestPrismDispatchQueue:
    
    @pytest.fixture
    def sample_data(self):
        return [
            ("drug1", "cell1", 1.5),
            ("drug2", "cell2", 2.0),
            ("drug3", "cell3", 0.5),
            ("drug1", "cell2", 1.8),
        ]
    
    @pytest.fixture
    def fake_lookup(self, sample_data):
        return FakePrismLookup(sample_data)
    
    def test_basic_initialization(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        assert queue.total == 4
        assert queue.remaining == 4
        assert queue.index == 0
        assert queue.has_next()
    
    def test_dispatch_order(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        
        # Dispatch all items and verify order matches lookup.keys()
        dispatched = []
        while queue.has_next():
            item = queue.dispatch()
            dispatched.append((item.drug, item.cell))
        
        assert dispatched == fake_lookup.keys()
        assert queue.remaining == 0
        assert not queue.has_next()
    
    def test_dispatch_item_structure(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        item = queue.dispatch()
        
        assert isinstance(item, DispatchItem)
        assert item.drug == "drug1"
        assert item.cell == "cell1"
        assert item.ic50 == 1.5
        assert isinstance(item.row, pd.Series)
        assert item.row["ic50"] == 1.5
        assert item.row["drug_name"] == "drug1"
    
    def test_peek_without_advancing(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        
        # Peek should return same item multiple times
        item1 = queue.peek()
        item2 = queue.peek()
        assert item1.drug == item2.drug == "drug1"
        assert queue.index == 0
        
        # Dispatch should return the same item
        item3 = queue.dispatch()
        assert item3.drug == "drug1"
        assert queue.index == 1
    
    def test_custom_order(self, fake_lookup):
        custom_order = [("drug3", "cell3"), ("drug1", "cell1")]
        queue = PrismDispatchQueue(fake_lookup, order=custom_order)
        
        assert queue.total == 2
        item1 = queue.dispatch()
        assert (item1.drug, item1.cell) == ("drug3", "cell3")
        
        item2 = queue.dispatch()
        assert (item2.drug, item2.cell) == ("drug1", "cell1")
        
        assert not queue.has_next()
    
    def test_custom_order_with_unknown_keys(self, fake_lookup):
        custom_order = [("unknown_drug", "unknown_cell"), ("drug1", "cell1")]
        
        with pytest.raises(
            KeyError, 
            match="These \\(drug, cell\\) keys are not in the lookup"
        ):
            PrismDispatchQueue(fake_lookup, order=custom_order)
    
    def test_shuffle_with_seed(self, fake_lookup):
        # Create two queues with same seed
        queue1 = PrismDispatchQueue(fake_lookup, shuffle=True, seed=42)
        queue2 = PrismDispatchQueue(fake_lookup, shuffle=True, seed=42)
        
        # Should have same order
        keys1 = queue1.keys
        keys2 = queue2.keys
        assert keys1 == keys2
        
        # Should be different from original order
        original_keys = fake_lookup.keys()
        assert keys1 != original_keys  # Very likely with 4 items
    
    def test_shuffle_different_seeds(self, fake_lookup):
        queue1 = PrismDispatchQueue(fake_lookup, shuffle=True, seed=42)
        queue2 = PrismDispatchQueue(fake_lookup, shuffle=True, seed=123)
        
        # Should likely have different orders
        keys1 = queue1.keys
        keys2 = queue2.keys
        # This could occasionally fail due to randomness, but very unlikely
        assert keys1 != keys2
    
    def test_reset_without_shuffle(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        
        # Dispatch some items
        queue.dispatch()
        queue.dispatch()
        assert queue.index == 2
        
        # Reset
        queue.reset(shuffle=False)
        assert queue.index == 0
        assert queue.has_next()
        
        # Should dispatch same first item
        item = queue.dispatch()
        assert (item.drug, item.cell) == fake_lookup.keys()[0]
    
    def test_reset_with_shuffle(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup, shuffle=False)
        original_keys = queue.keys
        
        # Reset with shuffle
        queue.reset(shuffle=True, seed=42)
        assert queue.index == 0
        shuffled_keys = queue.keys
        
        # Should be shuffled (very likely different)
        assert shuffled_keys != original_keys
    
    def test_empty_dispatch(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        
        # Exhaust the queue
        while queue.has_next():
            queue.dispatch()
        
        # Further dispatches should return None
        assert queue.dispatch() is None
        assert queue.peek() is None
    
    def test_state_serialization(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup, shuffle=True, seed=42)
        
        # Dispatch some items
        queue.dispatch()
        queue.dispatch()
        
        # Get state
        state = queue.to_state()
        
        assert "keys" in state
        assert "cursor" in state
        assert "seed" in state
        assert "shuffled" in state
        assert state["cursor"] == 2
        assert state["seed"] == 42
        assert state["shuffled"] is True
    
    def test_state_restoration(self, fake_lookup):
        # Create original queue
        original_queue = PrismDispatchQueue(fake_lookup, shuffle=True, seed=42)
        original_queue.dispatch()  # Advance cursor
        
        # Save and restore state
        state = original_queue.to_state()
        restored_queue = PrismDispatchQueue.from_state(fake_lookup, state)
        
        # Should have same state
        assert restored_queue.index == original_queue.index
        assert restored_queue.keys == original_queue.keys
        assert restored_queue.total == original_queue.total
        assert restored_queue.remaining == original_queue.remaining
    
    def test_state_restoration_invalid_cursor(self, fake_lookup):
        state = {
            "keys": fake_lookup.keys(),
            "cursor": 999,  # Invalid cursor
            "seed": None,
            "shuffled": False
        }
        
        with pytest.raises(ValueError, match="Invalid cursor"):
            PrismDispatchQueue.from_state(fake_lookup, state)
    
    def test_state_restoration_unknown_keys(self, fake_lookup):
        state = {
            "keys": [("unknown", "key"), ("drug1", "cell1")],
            "cursor": 0,
            "seed": None,
            "shuffled": False
        }
        
        with pytest.raises(
            KeyError, 
            match="These \\(drug, cell\\) keys are not in the lookup"
        ):
            PrismDispatchQueue.from_state(fake_lookup, state)
    
    def test_properties_during_dispatch(self, fake_lookup):
        queue = PrismDispatchQueue(fake_lookup)
        total = queue.total
        
        assert queue.index == 0
        assert queue.remaining == total
        
        # Dispatch one item
        queue.dispatch()
        assert queue.index == 1
        assert queue.remaining == total - 1
        
        # Dispatch all remaining
        while queue.has_next():
            queue.dispatch()
        
        assert queue.index == total
        assert queue.remaining == 0
    
    def test_empty_lookup(self):
        empty_lookup = FakePrismLookup([])
        queue = PrismDispatchQueue(empty_lookup)
        
        assert queue.total == 0
        assert queue.remaining == 0
        assert not queue.has_next()
        assert queue.dispatch() is None
        assert queue.peek() is None
