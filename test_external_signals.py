#!/usr/bin/env python3
"""
Test script for external signal functionality
This script simulates external signals being written to the signals.json file
and verifies that the Python bot can process them correctly.
"""
import json
import time
import asyncio
from pathlib import Path
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from trading.external_signal_handler import ExternalSignalHandler, ExternalSignal
from utils.config import Config
from broker.deriv import DerivAPIWrapper


def simulate_mt5_signal(symbol="R_100", direction="CALL", expiry_seconds=60):
    """Simulate an MT5 signal being written to the file"""
    signal_data = {
        "symbol": symbol,
        "direction": direction,
        "timestamp": time.time(),
        "expiry_seconds": expiry_seconds,
        "confidence": 0.7,
        "strategy": "test_ma_crossover"
    }
    
    signals_dir = Path("signals")
    signals_dir.mkdir(exist_ok=True)
    signal_file = signals_dir / "signals.json"
    
    # Read existing signals
    existing_signals = []
    if signal_file.exists():
        try:
            content = signal_file.read_text()
            if content.strip():
                existing_signals = json.loads(content)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in signal file, starting fresh")
            existing_signals = []
    
    # Add new signal
    existing_signals.append(signal_data)
    
    # Write back to file
    signal_file.write_text(json.dumps(existing_signals))
    print(f"Signal written to {signal_file}: {signal_data}")


def test_signal_validation():
    """Test the signal validation logic"""
    print("\n=== Testing Signal Validation ===")
    
    # Valid signal
    valid_signal = ExternalSignal(
        symbol="R_100",
        direction="CALL",
        timestamp=time.time(),
        expiry_seconds=60,
        confidence=0.7
    )
    assert valid_signal.is_valid(), "Valid signal should pass validation"
    print("✓ Valid signal passed validation")
    
    # Invalid direction
    invalid_direction = ExternalSignal(
        symbol="R_100",
        direction="INVALID",
        timestamp=time.time(),
        expiry_seconds=60
    )
    assert not invalid_direction.is_valid(), "Invalid direction should fail validation"
    print("✓ Invalid direction failed validation")
    
    # Invalid expiry
    invalid_expiry = ExternalSignal(
        symbol="R_100",
        direction="CALL",
        timestamp=time.time(),
        expiry_seconds=-1
    )
    assert not invalid_expiry.is_valid(), "Invalid expiry should fail validation"
    print("✓ Invalid expiry failed validation")
    
    # Invalid confidence
    invalid_confidence = ExternalSignal(
        symbol="R_100",
        direction="CALL",
        timestamp=time.time(),
        expiry_seconds=60,
        confidence=1.5
    )
    assert not invalid_confidence.is_valid(), "Invalid confidence should fail validation"
    print("✓ Invalid confidence failed validation")
    
    # Stale signal
    stale_signal = ExternalSignal(
        symbol="R_100",
        direction="CALL",
        timestamp=time.time() - 60,  # 60 seconds ago
        expiry_seconds=60
    )
    assert stale_signal.is_stale(max_age_seconds=30), "Stale signal should be detected"
    print("✓ Stale signal detected")


def test_file_operations():
    """Test file read/write operations"""
    print("\n=== Testing File Operations ===")
    
    signals_dir = Path("signals")
    signals_dir.mkdir(exist_ok=True)
    signal_file = signals_dir / "signals.json"
    
    # Clear the file first
    signal_file.write_text(json.dumps([]))
    
    # Test writing a signal
    simulate_mt5_signal("R_100", "CALL", 60)
    
    # Verify it was written
    content = signal_file.read_text()
    signals = json.loads(content)
    assert len(signals) == 1, "Signal should be written to file"
    assert signals[0]["symbol"] == "R_100", "Symbol should match"
    assert signals[0]["direction"] == "CALL", "Direction should match"
    print("✓ Signal written to file successfully")
    
    # Add another signal
    simulate_mt5_signal("EURUSD", "PUT", 120)
    
    # Verify both signals exist
    content = signal_file.read_text()
    signals = json.loads(content)
    assert len(signals) == 2, "Both signals should exist in file"
    print("✓ Multiple signals handled correctly")


async def test_integration_with_config():
    """Test integration with actual config and API setup"""
    print("\n=== Testing Integration ===")
    
    # Create a minimal config for testing
    # We'll use a mock config since we don't have real API credentials for testing
    import tempfile
    import os
    
    # Create a temporary env file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
APP_ID=test
API_TOKEN=test
LLM_API_TOKEN=test
ASSET=R_100
EXPIRY=1
STAKE=1.0
MAX_STAKE=10.0
TIMEFRAME=1
BACKTEST=false
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
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
EXTERNAL_SIGNAL_INTERVAL=5
""")
        temp_env_path = f.name
    
    try:
        # Load config from temp file
        os.environ['ENV_FILE'] = temp_env_path
        cfg = Config(temp_env_path)
        
        print("✓ Configuration loaded successfully")
        print(f"  - Asset: {cfg.asset}")
        print(f"  - External signals enabled: {cfg.enable_external_signals}")
        print(f"  - Signal interval: {cfg.external_signal_interval}s")
        
    finally:
        # Clean up temp file
        os.unlink(temp_env_path)


def main():
    print("Testing External Signal Handler Functionality")
    print("=" * 50)
    
    # Run validation tests
    test_signal_validation()
    
    # Test file operations
    test_file_operations()
    
    # Test integration
    asyncio.run(test_integration_with_config())
    
    print("\n" + "=" * 50)
    print("All tests passed! 🎉")
    print("\nTo test with real MT4/MT5:")
    print("1. Compile and attach MT5_Signal_Indicator.mq5 to a chart")
    print("2. Compile and run MT5_Signal_EA.mq5 as an Expert Advisor")
    print("3. Set ENABLE_EXTERNAL_SIGNALS=true in your env file")
    print("4. Run the bot with: python main.py")
    print("5. Monitor the signals/signals.json file for generated signals")


if __name__ == "__main__":
    main()