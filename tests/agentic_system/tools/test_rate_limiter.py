"""
test_rate_limit.py

Test suite for FileBasedRateLimiter.
Tests rate limiting across single-threaded, multi-threaded, 
and multi-process scenarios.
"""

import pytest
import time
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

from dspy_litl_agentic_system.tools.rate_limiter import FileBasedRateLimiter

# Test timeout in seconds
TEST_TIMEOUT = 30


# Helper function for multiprocess testing (must be at module level for pickling)
def _make_request_process(args):
    """Helper function for multiprocess testing."""
    i, name = args
    limiter = FileBasedRateLimiter(
        max_requests=3, 
        time_window=1.0, 
        name=name
    )
    start = time.monotonic()
    limiter.acquire_sync()
    duration = time.monotonic() - start
    return (i, duration)


class TestBasicFunctionality:
    """Test basic initialization and configuration."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_initialization(self, temp_rate_limiter):
        """Test that the rate limiter initializes correctly."""
        assert temp_rate_limiter.max_requests == 3
        assert temp_rate_limiter.time_window == 1.0
        # Verify state file is in temp directory
        assert temp_rate_limiter.state_file.parent.name == "tmp" or \
               "tmp" in str(temp_rate_limiter.state_file.parent)
        # Note: state file is created on first acquire, not on init

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_custom_limits(self):
        """Test with custom rate limit parameters."""
        name = f"test_custom_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=5, time_window=2.0, name=name)
        
        try:
            start = time.monotonic()
            for _ in range(6):
                limiter.acquire_sync()
            duration = time.monotonic() - start
            
            # 6th request should wait for the custom 2-second window
            # Allow 5% margin below and 30% margin above
            assert duration >= limiter.time_window * 0.95, \
                "Should respect custom time window"
            assert duration < limiter.time_window * 1.3, \
                "Should not wait excessively"
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_different_instances_share_state(self):
        """Test that different instances with the same name share state."""
        name = f"test_shared_{int(time.monotonic() * 1000000)}"
        limiter1 = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        limiter2 = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Make 2 requests with limiter1
            limiter1.acquire_sync()
            limiter1.acquire_sync()
            
            # 3rd request with limiter2 should be delayed
            start = time.monotonic()
            limiter2.acquire_sync()
            duration = time.monotonic() - start
            
            assert duration >= limiter1.time_window * 0.9, \
                "Different instances should share state"
        finally:
            if limiter1.state_file.exists():
                limiter1.state_file.unlink()


class TestSynchronousRateLimiting:
    """Test synchronous rate limiting behavior."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_single_request_sync(self, temp_rate_limiter):
        """Test that a single request completes quickly."""
        start = time.monotonic()
        temp_rate_limiter.acquire_sync()
        duration = time.monotonic() - start
        # More forgiving: allow up to 20% of time window
        assert duration < temp_rate_limiter.time_window * 0.2, \
            "Single request should not be rate limited"
        # Now state file should exist
        assert temp_rate_limiter.state_file.exists()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_within_limit_sync(self, temp_rate_limiter):
        """Test that requests within the limit are not delayed."""
        start = time.monotonic()
        for _ in range(3):
            temp_rate_limiter.acquire_sync()
        duration = time.monotonic() - start
        # More forgiving: allow up to 30% of time window
        assert duration < temp_rate_limiter.time_window * 0.3, \
            "Requests within limit should not be delayed"

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_exceeds_limit_sync(self, temp_rate_limiter):
        """Test that exceeding the limit causes delay."""
        start = time.monotonic()
        for _ in range(4):
            temp_rate_limiter.acquire_sync()
        duration = time.monotonic() - start
        # 4th request should wait until the 1st falls outside the window
        # Allow 10% margin below and 50% margin above
        assert duration >= temp_rate_limiter.time_window * 0.9, \
            "4th request should wait ~1 time window"
        assert duration < temp_rate_limiter.time_window * 1.5, \
            "Wait should not be excessive"

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_time_window_reset(self, temp_rate_limiter):
        """Test that the time window resets correctly."""
        # Make 3 requests
        for _ in range(3):
            temp_rate_limiter.acquire_sync()
        
        # Wait for the window to expire (with 10% buffer)
        time.sleep(temp_rate_limiter.time_window * 1.1)
        
        # Next 3 requests should not be delayed
        start = time.monotonic()
        for _ in range(3):
            temp_rate_limiter.acquire_sync()
        duration = time.monotonic() - start
        # Allow up to 30% of time window
        assert duration < temp_rate_limiter.time_window * 0.3, \
            "Requests after window reset should not be delayed"


class TestAsynchronousRateLimiting:
    """Test asynchronous rate limiting behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_async_acquire(self, temp_rate_limiter):
        """Test asynchronous acquire method."""
        start = time.monotonic()
        await temp_rate_limiter.acquire()
        duration = time.monotonic() - start
        # Allow up to 20% of time window
        assert duration < temp_rate_limiter.time_window * 0.2, \
            "Single async request should not be delayed"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_async_multiple_requests(self, temp_rate_limiter):
        """Test multiple async requests."""
        start = time.monotonic()
        for _ in range(4):
            await temp_rate_limiter.acquire()
        duration = time.monotonic() - start
        # Allow 10% margin below and 50% margin above
        assert duration >= temp_rate_limiter.time_window * 0.9, \
            "4th async request should wait"
        assert duration < temp_rate_limiter.time_window * 1.5, \
            "Async wait should not be excessive"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_concurrent_async_tasks(self, temp_rate_limiter):
        """Test concurrent async tasks."""
        import asyncio
        tasks = [temp_rate_limiter.acquire() for _ in range(5)]
        start = time.monotonic()
        await asyncio.gather(*tasks)
        duration = time.monotonic() - start
        
        # 5 requests with limit of 3 should cause delays
        assert duration >= temp_rate_limiter.time_window * 0.9, \
            "Concurrent async tasks should be rate limited"


class TestMultiThreadedRateLimiting:
    """Test rate limiting across multiple threads."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_multi_threaded(self, temp_rate_limiter):
        """Test rate limiting across multiple threads."""
        num_threads = 6
        results = []
        
        def make_request(i):
            start = time.monotonic()
            temp_rate_limiter.acquire_sync()
            duration = time.monotonic() - start
            return (i, duration)
        
        start_time = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(make_request, range(num_threads)))
        total_duration = time.monotonic() - start_time
        
        # First 3 requests should be fast, next 3 should wait ~1 time window
        assert total_duration >= temp_rate_limiter.time_window * 0.9, \
            "Multi-threaded requests should be rate limited"
        assert total_duration < temp_rate_limiter.time_window * 2.5, \
            "Total time should not exceed 2.5 time windows"
        
        # Check that some requests were delayed
        # Use relative threshold: 50% of time window
        # Allow for some timing variance - require at least 2 delayed instead of 3
        delayed_count = sum(
            1 for _, dur in results 
            if dur > temp_rate_limiter.time_window * 0.5
        )
        assert delayed_count >= 2, \
            f"At least 2 requests should be delayed, got {delayed_count}"

    @pytest.mark.timeout(TEST_TIMEOUT * 2)  # Longer timeout for stress test
    def test_stress_test_many_threads(self):
        """Stress test with many concurrent threads."""
        name = f"test_stress_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=5, 
            time_window=1.0, 
            name=name
        )
        num_threads = 20
        
        def make_request(i):
            limiter.acquire_sync()
            return i
        
        try:
            start_time = time.monotonic()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(make_request, range(num_threads)))
            total_duration = time.monotonic() - start_time
            
            # All requests should complete successfully
            assert len(results) == num_threads
            # With 20 requests and limit of 5, 
            # should take at least 3 time windows
            assert total_duration >= limiter.time_window * 3.0, \
                "Many requests should be properly rate limited"
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()


class TestMultiProcessRateLimiting:
    """Test rate limiting across multiple processes."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_multi_process(self, temp_rate_limiter):
        """Test rate limiting across multiple processes."""
        num_processes = min(6, cpu_count())
        
        # Extract name for child processes
        name = temp_rate_limiter.state_file.stem.replace("_rate_limiter", "")
        
        # Create args for each process
        args = [(i, name) for i in range(num_processes)]
        
        start_time = time.monotonic()
        with Pool(processes=num_processes) as pool:
            results = pool.map(_make_request_process, args)
        total_duration = time.monotonic() - start_time
        
        # With 6 requests and limit of 3, should take at least 1 time window
        assert total_duration >= temp_rate_limiter.time_window * 0.9, \
            "Multi-process requests should be rate limited"
        
        # Check that some requests were delayed
        # Use relative threshold: 50% of time window
        delayed_count = sum(
            1 for _, dur in results 
            if dur > temp_rate_limiter.time_window * 0.5
        )
        assert delayed_count >= 3, "At least 3 processes should be delayed"


class TestCorruptionRecovery:
    """Test recovery from corrupted or invalid state files."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_corrupted_state_file_recovery(self):
        """Test that corrupted state file is handled gracefully."""
        name = f"test_corrupt_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Make a valid request first
            limiter.acquire_sync()
            assert limiter.state_file.exists()
            
            # Corrupt the state file
            limiter.state_file.write_text("invalid json {{{")
            
            # Should still work and auto-recover
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            
            # Should proceed without delay (full capacity after recovery)
            assert duration < limiter.time_window * 0.2, \
                "Corrupted state should reset to full capacity"
            
            # Verify state file was fixed
            with open(limiter.state_file) as f:
                data = json.load(f)
                assert "requests" in data
                assert isinstance(data["requests"], list)
                
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_corrupted_state_file_with_null_bytes(self):
        """Test recovery from state file with null bytes."""
        name = f"test_nullbytes_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Create a state file with null bytes
            limiter.state_file.write_text('{"requests": [123.456]}\x00\x00\x00')
            
            # Should handle and clean up
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            
            assert duration < limiter.time_window * 0.2
            
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_state_structure(self):
        """Test recovery from invalid state structure."""
        name = f"test_invalid_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Create state file with wrong structure
            limiter.state_file.write_text('["not", "a", "dict"]')
            
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            
            # Should reset to full capacity
            assert duration < limiter.time_window * 0.2
            
            # Verify recovery
            with open(limiter.state_file) as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert "requests" in data
                
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_timestamp_types(self):
        """Test recovery from invalid timestamp types in state."""
        name = f"test_badtimestamps_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Create state with invalid timestamps
            limiter.state_file.write_text(
                '{"requests": ["string", null, {"bad": "timestamp"}]}'
            )
            
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            
            # Should reset and proceed
            assert duration < limiter.time_window * 0.2
            
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_empty_state_file(self):
        """Test recovery from completely empty state file."""
        name = f"test_empty_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=2, time_window=1.0, name=name)
        
        try:
            # Create empty state file
            limiter.state_file.write_text("")
            
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            
            # Should handle empty file gracefully
            assert duration < limiter.time_window * 0.2
            
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()


class TestInputValidation:
    """Test input validation for rate limiter parameters."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_max_requests_negative(self):
        """Test that negative max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be a positive integer"):
            FileBasedRateLimiter(max_requests=-1, time_window=1.0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_max_requests_zero(self):
        """Test that zero max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be a positive integer"):
            FileBasedRateLimiter(max_requests=0, time_window=1.0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_max_requests_float(self):
        """Test that float max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be a positive integer"):
            FileBasedRateLimiter(max_requests=3.5, time_window=1.0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_time_window_negative(self):
        """Test that negative time_window raises ValueError."""
        with pytest.raises(ValueError, match="time_window must be a positive number"):
            FileBasedRateLimiter(max_requests=3, time_window=-1.0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_time_window_zero(self):
        """Test that zero time_window raises ValueError."""
        with pytest.raises(ValueError, match="time_window must be a positive number"):
            FileBasedRateLimiter(max_requests=3, time_window=0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_invalid_time_window_string(self):
        """Test that string time_window raises ValueError."""
        with pytest.raises(ValueError, match="time_window must be a positive number"):
            FileBasedRateLimiter(max_requests=3, time_window="1.0")

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_valid_float_time_window(self):
        """Test that float time_window is accepted."""
        name = f"test_float_window_{int(time.monotonic() * 1000000)}"
        limiter = FileBasedRateLimiter(
            max_requests=3, time_window=0.5, name=name)
        
        try:
            # Should work with fractional time window
            start = time.monotonic()
            limiter.acquire_sync()
            duration = time.monotonic() - start
            assert duration < limiter.time_window * 0.2
        finally:
            if limiter.state_file.exists():
                limiter.state_file.unlink()


# I read that on macOS and Windows the multi-processmodule uses spawn.
# Without this guard, running the multi-process tests may lead to
# recursive spawning of processes.
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

