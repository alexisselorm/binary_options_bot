# MT4/MT5 to Python Binary Options Bot Bridge

## Overview

This implementation creates a bridge between MT4/MT5 trading platforms and the Python binary options bot, allowing signals generated in MT4/MT5 to trigger trades executed by the Python bot using the Deriv API.

## Architecture

```
MT5 Indicator → MT5 Expert Advisor → signals/signals.json → Python Signal Handler → Deriv API
```

## Components

### 1. MT5 Signal Indicator (`MT5_Signal_Indicator.mq5`)
- Generates trading signals based on technical analysis (MA crossover in this example)
- Uses indicator buffers to store signal values
- Writes signals to JSON file when new confirmed signals appear

### 2. MT5 Expert Advisor (`MT5_Signal_EA.mq5`)
- Monitors the indicator buffers for new signals
- Prevents duplicate signals by checking against previous bars
- Writes validated signals to the shared JSON file

### 3. Python Signal Handler (`trading/external_signal_handler.py`)
- Monitors the signals JSON file for new entries
- Validates incoming signals (format, timestamps, etc.)
- Executes trades using the existing bot infrastructure
- Integrates with risk management and logging systems

### 4. Modified Core Files
- `main.py`: Initializes and starts the external signal handler
- `utils/config.py`: Added external signal configuration options
- `trading/executor.py`: Conditionally disables internal signal generation when external signals are enabled

## Signal Format

```json
{
  "symbol": "EURUSD",
  "direction": "CALL" | "PUT",
  "timestamp": 1700000000,
  "expiry_seconds": 60,
  "confidence": 0.0-1.0 (optional),
  "strategy": "string" (optional)
}
```

## Configuration

Add these settings to your `.env` file:

```bash
# Enable external signals from MT4/MT5
ENABLE_EXTERNAL_SIGNALS=true

# Interval in seconds to check for new signals
EXTERNAL_SIGNAL_INTERVAL=5
```

## Setup Instructions

### MT5 Side:
1. Compile `MT5_Signal_Indicator.mq5` and attach it to a chart
2. Compile `MT5_Signal_EA.mq5` and attach it as an Expert Advisor to the same chart
3. Ensure both have write permissions to the data folder

### Python Bot Side:
1. Set `ENABLE_EXTERNAL_SIGNALS=true` in your `.env` file
2. Run the bot normally: `python main.py`
3. The bot will now listen for external signals instead of generating internal ones

## Features

- **Duplicate Prevention**: Tracks processed signals to prevent duplicate trades
- **Stale Signal Detection**: Ignores signals older than 30 seconds
- **Validation**: Checks signal format and values before processing
- **Integration**: Uses existing risk management, logging, and execution logic
- **Clean Shutdown**: Properly stops monitoring when the bot shuts down

## Risk Management

The system inherits all existing risk management features:
- Maximum consecutive losses limits
- Minimum balance thresholds
- Stake sizing based on risk percentage
- Martingale controls (if enabled)

## Testing

Run the test suite to verify functionality:
```bash
python test_external_signals.py
```

## Known Limitations

1. **File-based Communication**: Relies on file system access between MT4/MT5 and Python environments
2. **Timing**: Small delay between signal generation and execution due to file polling
3. **Platform Dependency**: Requires MT4/MT5 to be running on the same machine or with shared file access

## Future Improvements

1. **Network Communication**: Implement ZeroMQ or WebSocket for faster communication
2. **Enhanced Validation**: Add more sophisticated signal validation
3. **Multiple Sources**: Support signals from multiple MT4/MT5 instances
4. **Performance Monitoring**: Track signal accuracy and profitability by source