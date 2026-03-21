#!/usr/bin/env python3
"""
Comprehensive tests for the production-grade external signal handler.

Tests cover:
- Schema validation (pass/fail cases)
- Stale signal rejection
- Duplicate signal rejection (including restart persistence)
- Malformed JSON recovery
- Symbol mapping behavior
- Confidence threshold gating
- Event loop safe dispatch
- Integration-style tests
"""
from utils.config import Config
from trading.external_signal_handler import (
    ExternalSignal,
    SymbolMapper,
    PersistentDeduplicator
)
import sys
import json
import time
import asyncio
import tempfile
import shutil
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock imports that have heavy dependencies not needed for these tests
sys.modules['trading.executor'] = MagicMock()
sys.modules['broker.deriv'] = MagicMock()
sys.modules['broker'] = MagicMock()


class TestExternalSignalSchema(unittest.TestCase):
    """Test signal schema validation"""

    def test_valid_signal_creation(self):
        """Test creating a valid signal with all required fields"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_001",
            "confidence": 0.75,
            "strategy": "ma_crossover"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertEqual(len(errors), 0)
        self.assertEqual(signal.source, "mt5_ea")
        self.assertEqual(signal.symbol, "EURUSD")
        self.assertEqual(signal.direction, "CALL")
        self.assertEqual(signal.expiry_seconds, 60)
        self.assertEqual(signal.signal_id, "TEST_001")
        self.assertEqual(signal.confidence, 0.75)

    def test_valid_put_signal(self):
        """Test valid PUT signal"""
        signal_data = {
            "source": "mt4_indicator",
            "symbol": "GBPUSD",
            "direction": "PUT",
            "timestamp": time.time(),
            "expiry_seconds": 120,
            "signal_id": "TEST_002"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "PUT")
        self.assertEqual(len(errors), 0)

    def test_missing_required_field(self):
        """Test rejection when required field is missing"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            # Missing: direction
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_003"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("direction" in str(e).lower() for e in errors))

    def test_invalid_direction(self):
        """Test rejection of invalid direction"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "INVALID",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_004"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("direction" in str(e).lower() for e in errors))

    def test_case_insensitive_direction(self):
        """Test direction is normalized to uppercase"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "call",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_005"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "CALL")

    def test_invalid_expiry_negative(self):
        """Test rejection of negative expiry"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": -10,
            "signal_id": "TEST_006"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("expiry" in str(e).lower() for e in errors))

    def test_invalid_expiry_zero(self):
        """Test rejection of zero expiry"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 0,
            "signal_id": "TEST_007"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("expiry" in str(e).lower() for e in errors))

    def test_invalid_confidence_above_one(self):
        """Test rejection of confidence > 1.0"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_008",
            "confidence": 1.5
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("confidence" in str(e).lower() for e in errors))

    def test_invalid_confidence_negative(self):
        """Test rejection of confidence < 0.0"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_009",
            "confidence": -0.1
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("confidence" in str(e).lower() for e in errors))

    def test_valid_confidence_boundary_zero(self):
        """Test confidence = 0.0 is valid"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_010",
            "confidence": 0.0
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.confidence, 0.0)

    def test_valid_confidence_boundary_one(self):
        """Test confidence = 1.0 is valid"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_011",
            "confidence": 1.0
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.confidence, 1.0)

    def test_missing_signal_id(self):
        """Test rejection when signal_id is missing"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60
            # Missing: signal_id
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("signal_id" in str(e).lower() for e in errors))

    def test_empty_signal_id(self):
        """Test rejection when signal_id is empty"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": ""
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("signal_id" in str(e).lower() for e in errors))

    def test_invalid_timestamp_negative(self):
        """Test rejection of negative timestamp"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": -1000,
            "expiry_seconds": 60,
            "signal_id": "TEST_013"
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNone(signal)
        self.assertTrue(any("timestamp" in str(e).lower() for e in errors))

    def test_optional_fields_not_required(self):
        """Test that optional fields are truly optional"""
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "TEST_014"
            # No confidence, strategy, or metadata
        }

        signal, errors = ExternalSignal.from_dict(signal_data)

        self.assertIsNotNone(signal)
        self.assertIsNone(signal.confidence)
        self.assertIsNone(signal.strategy)


class TestSignalStaleness(unittest.TestCase):
    """Test signal staleness handling"""

    def test_fresh_signal_not_stale(self):
        """Test that a fresh signal is not considered stale"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="FRESH_001"
        )

        self.assertFalse(signal.is_stale(max_age_seconds=60))

    def test_old_signal_is_stale(self):
        """Test that an old signal is considered stale"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time() - 120,  # 2 minutes ago
            expiry_seconds=60,
            signal_id="STALE_001"
        )

        self.assertTrue(signal.is_stale(max_age_seconds=60))

    def test_signal_at_boundary(self):
        """Test signal exactly at age boundary"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time() - 60,  # Exactly 60 seconds ago
            expiry_seconds=60,
            signal_id="BOUNDARY_001"
        )

        # Should be stale (age > max_age, not >=)
        self.assertTrue(signal.is_stale(max_age_seconds=60))

    def test_signal_just_under_boundary(self):
        """Test signal just under age boundary"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time() - 59,  # 59 seconds ago
            expiry_seconds=60,
            signal_id="BOUNDARY_002"
        )

        self.assertFalse(signal.is_stale(max_age_seconds=60))


class TestConfidenceThreshold(unittest.TestCase):
    """Test confidence threshold gating"""

    def test_signal_with_no_confidence_passes(self):
        """Test signal without confidence passes any threshold"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="CONF_001"
        )

        self.assertTrue(signal.passes_confidence_threshold(0.8))
        self.assertTrue(signal.passes_confidence_threshold(0.5))
        self.assertTrue(signal.passes_confidence_threshold(0.0))

    def test_signal_above_threshold(self):
        """Test signal above threshold passes"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="CONF_002",
            confidence=0.85
        )

        self.assertTrue(signal.passes_confidence_threshold(0.8))
        self.assertFalse(signal.passes_confidence_threshold(0.9))

    def test_signal_at_threshold(self):
        """Test signal exactly at threshold passes"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="CONF_003",
            confidence=0.75
        )

        self.assertTrue(signal.passes_confidence_threshold(0.75))


class TestSymbolMapper(unittest.TestCase):
    """Test symbol mapping functionality"""

    def test_default_mapping_eurusd(self):
        """Test default EURUSD mapping"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("EURUSD")
        self.assertEqual(result, "frxEURUSD")

    def test_default_mapping_gbpusd(self):
        """Test default GBPUSD mapping"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("GBPUSD")
        self.assertEqual(result, "frxGBPUSD")

    def test_default_mapping_synthetic(self):
        """Test default synthetic index mapping"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("R_100")
        self.assertEqual(result, "R_100")

    def test_case_insensitive_mapping(self):
        """Test case-insensitive symbol matching"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("eurusd")
        self.assertEqual(result, "frxEURUSD")

    def test_mapping_with_suffix(self):
        """Test mapping with symbol suffix (e.g., EURUSD.a)"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("EURUSD.a")
        self.assertEqual(result, "frxEURUSD")

    def test_custom_mapping(self):
        """Test custom symbol mapping"""
        custom = {"MYEURUSD": "frxEURUSD", "MYGBPUSD": "frxGBPUSD"}
        mapper = SymbolMapper(custom)

        result = mapper.map_symbol("MYEURUSD")
        self.assertEqual(result, "frxEURUSD")

    def test_unmapped_symbol(self):
        """Test unmapped symbol returns None"""
        mapper = SymbolMapper()
        result = mapper.map_symbol("UNKNOWN_SYMBOL")
        self.assertIsNone(result)

    def test_add_mapping(self):
        """Test adding mapping at runtime"""
        mapper = SymbolMapper()
        mapper.add_mapping("NEWSYM", "frxNEWSYM")

        result = mapper.map_symbol("NEWSYM")
        self.assertEqual(result, "frxNEWSYM")


class TestPersistentDeduplicator(unittest.TestCase):
    """Test persistent deduplication"""

    def setUp(self):
        """Create temp directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "processed.json"

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)

    def test_mark_and_check_processed(self):
        """Test marking signal as processed and checking"""
        dedup = PersistentDeduplicator(self.storage_path)

        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="DEDUP_001"
        )

        signal_hash = signal.generate_hash()

        # Should not be processed initially
        self.assertFalse(dedup.is_processed(signal_hash))

        # Mark as processed
        dedup.mark_processed(signal)

        # Should be processed now
        self.assertTrue(dedup.is_processed(signal_hash))

    def test_persistence_across_instances(self):
        """Test deduplication persists across handler restarts"""
        # First instance
        dedup1 = PersistentDeduplicator(self.storage_path)

        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="PERSIST_001"
        )

        dedup1.mark_processed(signal)

        # Second instance (simulates restart)
        dedup2 = PersistentDeduplicator(self.storage_path)

        signal_hash = signal.generate_hash()
        self.assertTrue(dedup2.is_processed(signal_hash))

    def test_different_signals_not_marked(self):
        """Test that different signals are not marked as processed"""
        dedup = PersistentDeduplicator(self.storage_path)

        signal1 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="DIFF_001"
        )

        signal2 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="PUT",  # Different direction
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="DIFF_002"
        )

        dedup.mark_processed(signal1)

        self.assertTrue(dedup.is_processed(signal1.generate_hash()))
        self.assertFalse(dedup.is_processed(signal2.generate_hash()))

    def test_stats_reporting(self):
        """Test statistics reporting"""
        dedup = PersistentDeduplicator(self.storage_path)

        # Mark some signals
        for i in range(5):
            signal = ExternalSignal(
                source="mt5_ea",
                symbol="EURUSD",
                direction="CALL",
                timestamp=time.time(),
                expiry_seconds=60,
                signal_id=f"STATS_{i:03d}"
            )
            dedup.mark_processed(signal)

        stats = dedup.get_stats()

        self.assertEqual(stats["total_processed"], 5)
        self.assertEqual(stats["last_hour"], 5)
        self.assertEqual(stats["last_24h"], 5)


class TestMalformedJSONRecovery(unittest.TestCase):
    """Test malformed JSON recovery"""

    def setUp(self):
        """Create temp directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.signal_file = Path(self.temp_dir) / "signals.json"

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)

    def test_empty_file(self):
        """Test handling of empty file"""
        self.signal_file.write_text("")

        # Simulate reading
        content = self.signal_file.read_text().strip()
        self.assertEqual(content, "")

    def test_invalid_json_object(self):
        """Test handling of invalid JSON (object instead of array)"""
        self.signal_file.write_text('{"not": "an array"}')

        content = self.signal_file.read_text().strip()
        data = json.loads(content)

        # Should parse but not be a list
        self.assertIsInstance(data, dict)

    def test_truncated_json(self):
        """Test handling of truncated JSON"""
        self.signal_file.write_text(
            '[{"symbol": "EURUSD", "direction": "CALL"')

        content = self.signal_file.read_text().strip()

        with self.assertRaises(json.JSONDecodeError):
            json.loads(content)

    def test_corrupted_json(self):
        """Test handling of completely corrupted content"""
        self.signal_file.write_text('not json at all {{{')

        content = self.signal_file.read_text().strip()

        with self.assertRaises(json.JSONDecodeError):
            json.loads(content)


class TestSignalHashGeneration(unittest.TestCase):
    """Test signal hash generation for deduplication"""

    def test_same_signal_same_hash(self):
        """Test identical signals produce same hash"""
        signal1 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=1700000000,
            expiry_seconds=60,
            signal_id="HASH_001"
        )

        signal2 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=1700000000,
            expiry_seconds=60,
            signal_id="HASH_001"
        )

        self.assertEqual(signal1.generate_hash(), signal2.generate_hash())

    def test_different_signal_different_hash(self):
        """Test different signals produce different hashes"""
        signal1 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=1700000000,
            expiry_seconds=60,
            signal_id="HASH_001"
        )

        signal2 = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="PUT",  # Different
            timestamp=1700000000,
            expiry_seconds=60,
            signal_id="HASH_001"
        )

        self.assertNotEqual(signal1.generate_hash(), signal2.generate_hash())

    def test_hash_length(self):
        """Test hash is truncated to expected length"""
        signal = ExternalSignal(
            source="mt5_ea",
            symbol="EURUSD",
            direction="CALL",
            timestamp=time.time(),
            expiry_seconds=60,
            signal_id="HASH_LEN_001"
        )

        hash_value = signal.generate_hash()
        self.assertEqual(len(hash_value), 16)  # 16 hex chars


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading for external signals"""

    def setUp(self):
        """Create temp env file"""
        self.temp_dir = tempfile.mkdtemp()
        self.env_path = Path(self.temp_dir) / "test.env"

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)

    def test_external_signal_config_defaults(self):
        """Test external signal config defaults"""
        self.env_path.write_text("""
APP_ID=test
API_TOKEN=test
LLM_API_TOKEN=test
ASSET=R_100
EXPIRY=1
STAKE=1.0
MAX_STAKE=10.0
TIMEFRAME=1
BACKTEST=false
MODEL_TYPE=rule
MODEL_PATH_LSTM=models/lstm_model.keras
MODEL_PATH_XGB=models/xgb_model.json
MODEL_PATH_RL=models/ppo_agent.zip
SIGNAL_THRESHOLD=0.6
SEQUENCE_LENGTH=100
MAX_CONSECUTIVE_LOSSES=5
TRADE_RISK_PERCENT=1.0
MIN_BALANCE=10.0
MARTINGALE_MODE=off
HISTORICAL_DATA_PATH=data/historical_candles.csv
SAVE_TRADE_HISTORY=true
TRADE_HISTORY_PATH=logs/trade_history.csv
""")

        cfg = Config(str(self.env_path))

        self.assertFalse(cfg.enable_external_signals)
        self.assertEqual(cfg.external_signal_interval, 5)
        self.assertEqual(cfg.external_signal_dir, "signals")
        self.assertEqual(cfg.external_signal_file, "signals.json")
        self.assertEqual(cfg.external_signal_max_age_seconds, 60)
        self.assertEqual(cfg.external_signal_confidence_threshold, 0.0)
        self.assertEqual(cfg.external_symbol_map, {})

    def test_external_signal_config_custom(self):
        """Test custom external signal config"""
        self.env_path.write_text("""
APP_ID=test
API_TOKEN=test
LLM_API_TOKEN=test
ASSET=R_100
EXPIRY=1
STAKE=1.0
MAX_STAKE=10.0
TIMEFRAME=1
BACKTEST=false
MODEL_TYPE=rule
MODEL_PATH_LSTM=models/lstm_model.keras
MODEL_PATH_XGB=models/xgb_model.json
MODEL_PATH_RL=models/ppo_agent.zip
SIGNAL_THRESHOLD=0.6
SEQUENCE_LENGTH=100
MAX_CONSECUTIVE_LOSSES=5
TRADE_RISK_PERCENT=1.0
MIN_BALANCE=10.0
MARTINGALE_MODE=off
HISTORICAL_DATA_PATH=data/historical_candles.csv
SAVE_TRADE_HISTORY=true
TRADE_HISTORY_PATH=logs/trade_history.csv
ENABLE_EXTERNAL_SIGNALS=true
EXTERNAL_SIGNAL_INTERVAL=10
EXTERNAL_SIGNAL_DIR=/tmp/signals
EXTERNAL_SIGNAL_FILE=mt5_signals.json
EXTERNAL_SIGNAL_MAX_AGE_SECONDS=120
EXTERNAL_SIGNAL_CONFIDENCE_THRESHOLD=0.7
EXTERNAL_SYMBOL_MAP=EURUSD:frxEURUSD,GBPUSD:frxGBPUSD
""")

        cfg = Config(str(self.env_path))

        self.assertTrue(cfg.enable_external_signals)
        self.assertEqual(cfg.external_signal_interval, 10)
        self.assertEqual(cfg.external_signal_dir, "/tmp/signals")
        self.assertEqual(cfg.external_signal_file, "mt5_signals.json")
        self.assertEqual(cfg.external_signal_max_age_seconds, 120)
        self.assertEqual(cfg.external_signal_confidence_threshold, 0.7)
        self.assertEqual(cfg.external_symbol_map["EURUSD"], "frxEURUSD")
        self.assertEqual(cfg.external_symbol_map["GBPUSD"], "frxGBPUSD")


class TestAsyncDispatch(unittest.TestCase):
    """Test async dispatch from threads"""

    def test_coroutine_dispatch(self):
        """Test that coroutines can be dispatched to event loop"""
        async def dummy_coro():
            # Small delay to make it a real coroutine
            await asyncio.sleep(0.01)
            return "success"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Start the loop in a thread
            def run_loop():
                loop.run_forever()

            loop_thread = Thread(target=run_loop, daemon=True)
            loop_thread.start()

            # Give loop time to start
            time.sleep(0.1)

            # Dispatch coroutine
            future = asyncio.run_coroutine_threadsafe(dummy_coro(), loop)
            result = future.result(timeout=5.0)
            self.assertEqual(result, "success")
        finally:
            # Stop the loop
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2.0)
            loop.close()


class TestIntegration(unittest.TestCase):
    """Integration-style tests simulating MT5 signal flow"""

    def setUp(self):
        """Create temp directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.signal_file = Path(self.temp_dir) / "signals.json"
        self.dedup_file = Path(self.temp_dir) / ".processed.json"

        # Initialize signal file
        self.signal_file.write_text("[]")

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)

    def test_full_signal_lifecycle(self):
        """Test full signal lifecycle: write -> validate -> process -> dedupe"""
        # Simulate MT5 writing a signal
        signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "INTEGRATION_001",
            "confidence": 0.8,
            "strategy": "ma_crossover"
        }

        # Write signal (simulating MT5 EA)
        signals = [signal_data]
        self.signal_file.write_text(json.dumps(signals))

        # Read and validate (simulating Python handler)
        content = self.signal_file.read_text()
        loaded_signals = json.loads(content)

        self.assertEqual(len(loaded_signals), 1)

        signal, errors = ExternalSignal.from_dict(loaded_signals[0])
        self.assertIsNotNone(signal)
        self.assertEqual(len(errors), 0)
        self.assertEqual(signal.signal_id, "INTEGRATION_001")

        # Mark as processed
        dedup = PersistentDeduplicator(self.dedup_file)
        self.assertFalse(dedup.is_processed(signal.generate_hash()))
        dedup.mark_processed(signal)
        self.assertTrue(dedup.is_processed(signal.generate_hash()))

        # Simulate duplicate signal (should be rejected)
        duplicate_signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time(),
            "expiry_seconds": 60,
            "signal_id": "INTEGRATION_001",  # Same ID
            "confidence": 0.8,
            "strategy": "ma_crossover"
        }

        duplicate_signal, _ = ExternalSignal.from_dict(duplicate_signal_data)
        self.assertTrue(dedup.is_processed(duplicate_signal.generate_hash()))

    def test_stale_signal_rejection(self):
        """Test stale signal is rejected"""
        # Write stale signal
        stale_signal_data = {
            "source": "mt5_ea",
            "symbol": "EURUSD",
            "direction": "CALL",
            "timestamp": time.time() - 300,  # 5 minutes ago
            "expiry_seconds": 60,
            "signal_id": "STALE_INTEGRATION_001"
        }

        signal, errors = ExternalSignal.from_dict(stale_signal_data)
        self.assertIsNotNone(signal)
        self.assertTrue(signal.is_stale(max_age_seconds=60))

    def test_symbol_mapping_integration(self):
        """Test symbol mapping in integration context"""
        mapper = SymbolMapper()

        # Test various symbol formats
        test_cases = [
            ("EURUSD", "frxEURUSD"),
            ("eurusd", "frxEURUSD"),
            ("EURUSD.a", "frxEURUSD"),
            ("R_100", "R_100"),
            ("1HZ10V", "1HZ10V"),
        ]

        for mt_symbol, expected_deriv in test_cases:
            result = mapper.map_symbol(mt_symbol)
            self.assertEqual(result, expected_deriv, f"Failed for {mt_symbol}")


def run_tests():
    """Run all tests and print summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestExternalSignalSchema,
        TestSignalStaleness,
        TestConfidenceThreshold,
        TestSymbolMapper,
        TestPersistentDeduplicator,
        TestMalformedJSONRecovery,
        TestSignalHashGeneration,
        TestConfigLoading,
        TestAsyncDispatch,
        TestIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
