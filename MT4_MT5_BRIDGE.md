# MT4/MT5 to Python Binary Options Bot Bridge

## Overview

This implementation creates a **production-grade bridge** between MT4/MT5 trading platforms and the Python binary options bot, allowing signals generated in MT4/MT5 to trigger trades executed by the Python bot using the Deriv API.

### Key Features

- **Strict Schema Validation**: All signals must conform to a defined schema with required fields
- **Atomic File Operations**: Prevents corruption from concurrent reads/writes
- **Persistent Deduplication**: Survives bot restarts using file-based storage
- **Symbol Mapping**: Configurable mapping from MT symbols to Deriv symbols
- **Confidence Threshold Gating**: Filter signals by minimum confidence
- **Stale Signal Rejection**: Automatically ignores old signals
- **Async-Safe Dispatch**: Proper thread-to-event-loop communication
- **Malformed JSON Recovery**: Graceful handling of corrupted signal files

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────┐
│ MT5 Indicator   │────▶│ MT5 Expert EA   │────▶│ signals.json     │────▶│ Python Handler  │────▶│ Deriv API   │
│ (Signal Gen)    │     │ (Signal Writer) │     │ (Shared File)    │     │ (Validator)     │     │ (Execution) │
└─────────────────┘     └─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────┘
                                                                                   │
                                                                                   ▼
                                                                          ┌──────────────────┐
                                                                          │ Risk Management  │
                                                                          │ Logging          │
                                                                          │ Deduplication    │
                                                                          └──────────────────┘
```

## Signal Schema

All signals must conform to this strict schema:

```json
{
  "source": "mt5_ea",           // Required: Signal origin identifier
  "symbol": "EURUSD",           // Required: MT4/MT5 symbol
  "direction": "CALL",          // Required: "CALL" or "PUT"
  "timestamp": 1700000000,      // Required: Unix timestamp (seconds)
  "expiry_seconds": 60,         // Required: Trade duration in seconds
  "signal_id": "MT5_EUR_CALL_1700000000_123",  // Required: Unique identifier
  "confidence": 0.75,           // Optional: Confidence score (0.0-1.0)
  "strategy": "ma_crossover",   // Optional: Strategy name
  "metadata": {}                // Optional: Additional data
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Origin of the signal (e.g., `mt5_ea`, `mt4_indicator`) |
| `symbol` | string | Yes | Trading symbol from MT platform |
| `direction` | string | Yes | Trade direction: `CALL` or `PUT` |
| `timestamp` | number | Yes | Unix timestamp in seconds when signal was generated |
| `expiry_seconds` | number | Yes | Trade duration in seconds |
| `signal_id` | string | Yes | Unique identifier for deduplication |
| `confidence` | number | No | Confidence score between 0.0 and 1.0 |
| `strategy` | string | No | Name of the strategy that generated the signal |
| `metadata` | object | No | Additional arbitrary data |

## Configuration

### Python Bot Configuration (`.env` file)

```bash
# ─────────────────────────────────────────────────────────────────────
# External Signal Settings
# ─────────────────────────────────────────────────────────────────────

# Enable/disable external signal processing
ENABLE_EXTERNAL_SIGNALS=true

# Directory containing signal files (relative to bot root)
EXTERNAL_SIGNAL_DIR=signals

# Name of the signal file
EXTERNAL_SIGNAL_FILE=signals.json

# Interval in seconds to check for new signals
EXTERNAL_SIGNAL_INTERVAL=5

# Maximum age of signals in seconds (older signals are rejected)
EXTERNAL_SIGNAL_MAX_AGE_SECONDS=60

# Minimum confidence threshold (0.0 = accept all, 1.0 = only highest)
EXTERNAL_SIGNAL_CONFIDENCE_THRESHOLD=0.0

# Symbol mapping from MT to Deriv symbols
# Format: "MT_SYMBOL:DERIV_SYMBOL,MT_SYMBOL2:DERIV_SYMBOL2"
# Or JSON format: '{"EURUSD":"frxEURUSD","GBPUSD":"frxGBPUSD"}'
EXTERNAL_SYMBOL_MAP=EURUSD:frxEURUSD,GBPUSD:frxGBPUSD,USDJPY:frxUSDJPY
```

### Default Symbol Mappings

The following mappings are built-in:

| MT Symbol | Deriv Symbol |
|-----------|--------------|
| EURUSD | frxEURUSD |
| GBPUSD | frxGBPUSD |
| USDJPY | frxUSDJPY |
| AUDUSD | frxAUDUSD |
| USDCAD | frxUSDCAD |
| USDCHF | frxUSDCHF |
| NZDUSD | frxNZDUSD |
| EURGBP | frxEURGBP |
| EURJPY | frxEURJPY |
| GBPJPY | frxGBPJPY |
| R_100 | R_100 |
| R_75 | R_75 |
| R_50 | R_50 |
| R_25 | R_25 |
| R_10 | R_10 |
| 1HZ10V | 1HZ10V |
| 1HZ100V | 1HZ100V |

## Setup Instructions

### Step 1: MT5 Terminal Configuration

#### 1.1 Locate MT5 Data Folder

1. Open MT5 terminal
2. Go to `File` → `Open Data Folder`
3. Note the path (e.g., `C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX`)

#### 1.2 Install Indicator and EA

1. Copy `MT5_Signal_Indicator.mq5` to:
   ```
   <MT5 Data Folder>\MQL5\Indicators\
   ```

2. Copy `MT5_Signal_EA.mq5` to:
   ```
   <MT5 Data Folder>\MQL5\Experts\
   ```

3. Restart MT5 or refresh the Navigator panel

#### 1.3 Compile Files

1. Open MetaEditor (F4 in MT5)
2. Navigate to `Indicators\MT5_Signal_Indicator.mq5`
3. Click Compile (F7)
4. Navigate to `Experts\MT5_Signal_EA.mq5`
5. Click Compile (F7)

Ensure both compile without errors.

#### 1.4 Attach to Chart

1. Open a chart for the symbol you want to trade (e.g., EURUSD)
2. Attach `MT5_Signal_Indicator` to the chart:
   - Navigator → Indicators → MT5_Signal_Indicator
   - Configure parameters as needed
3. Attach `MT5_Signal_EA` to the same chart:
   - Navigator → Expert Advisors → MT5_Signal_EA
   - Enable "Allow Algo Trading" (Alt+A)
   - Configure parameters:
     - `InpSymbol`: Trading symbol (e.g., EURUSD)
     - `InpSignalPeriod`: Signal expiry in seconds
     - `InpConfidence`: Signal confidence (0.0-1.0)
     - `InpStrategy`: Strategy name
     - `InpSignalSource`: Source identifier

### Step 2: Shared File Location

#### Windows (MT5 and Python on same machine)

The MT5 EA writes to the MT5 Files folder by default:
```
C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\MQL5\Files\signals.json
```

Configure the Python bot to read from this location:
```bash
EXTERNAL_SIGNAL_DIR=C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\MQL5\Files
EXTERNAL_SIGNAL_FILE=signals.json
```

#### Cross-Platform (MT5 on Windows, Python on Linux/Mac)

Use a shared folder or network mount:

**Option A: Shared Folder (Same Network)**
1. Create a shared folder accessible by both systems
2. Configure MT5 to write to the shared folder (modify EA)
3. Configure Python to read from the shared folder

**Option B: Cloud Sync Folder**
1. Use Dropbox, Google Drive, or similar
2. Point both MT5 and Python to the sync folder
3. Be aware of sync delays

**Option C: Network File Share**
1. Set up SMB/NFS share
2. Mount on both systems
3. Configure paths accordingly

### Step 3: Python Bot Configuration

1. Edit your `env` file (in the bot root directory)

2. Add/modify external signal settings:
   ```bash
   ENABLE_EXTERNAL_SIGNALS=true
   EXTERNAL_SIGNAL_INTERVAL=5
   EXTERNAL_SIGNAL_MAX_AGE_SECONDS=60
   EXTERNAL_SIGNAL_CONFIDENCE_THRESHOLD=0.0
   EXTERNAL_SYMBOL_MAP=EURUSD:frxEURUSD
   ```

3. Ensure other required settings are configured:
   ```bash
   APP_ID=your_deriv_app_id
   API_TOKEN=your_deriv_api_token
   ASSET=R_100  # Default asset (overridden by signal mapping)
   EXPIRY=1
   STAKE=10
   MIN_BALANCE=50
   ```

### Step 4: Run the Bot

```bash
python main.py
```

The bot will:
1. Initialize the external signal handler
2. Start monitoring the signal file
3. Wait for valid signals
4. Execute trades when signals are received

## Testing the Bridge

### Unit Tests

Run the comprehensive test suite:

```bash
python test_external_signals.py
```

This tests:
- Schema validation (pass/fail cases)
- Stale signal rejection
- Duplicate signal rejection
- Malformed JSON recovery
- Symbol mapping
- Confidence threshold gating
- Persistent deduplication

### Manual Testing

1. **Create a test signal file:**
   ```bash
   mkdir -p signals
   echo '[]' > signals/signals.json
   ```

2. **Write a test signal:**
   ```python
   import json
   import time
   
   signal = {
       "source": "test",
       "symbol": "EURUSD",
       "direction": "CALL",
       "timestamp": time.time(),
       "expiry_seconds": 60,
       "signal_id": f"TEST_{int(time.time())}",
       "confidence": 0.8,
       "strategy": "test_strategy"
   }
   
   with open("signals/signals.json", "w") as f:
       json.dump([signal], f)
   ```

3. **Monitor the bot logs** for signal processing

### Integration Test Script

```python
#!/usr/bin/env python3
"""Quick integration test for external signals"""
import json
import time
from pathlib import Path

signals_dir = Path("signals")
signals_dir.mkdir(exist_ok=True)
signal_file = signals_dir / "signals.json"

# Write test signal
signal = {
    "source": "integration_test",
    "symbol": "EURUSD",
    "direction": "CALL",
    "timestamp": time.time(),
    "expiry_seconds": 60,
    "signal_id": f"INT_TEST_{int(time.time())}",
    "confidence": 0.75,
    "strategy": "test"
}

# Read existing, add new, write back
existing = []
if signal_file.exists():
    try:
        existing = json.loads(signal_file.read_text())
    except:
        pass

existing.append(signal)
signal_file.write_text(json.dumps(existing))
print(f"Test signal written: {signal['signal_id']}")
```

## Troubleshooting

### Signals Not Being Processed

**Check 1: File Path**
```bash
# Verify signal file exists and is readable
ls -la signals/signals.json
```

**Check 2: File Permissions**
Ensure Python has read access to the signal file.

**Check 3: Signal Format**
Validate signal JSON format:
```python
import json
with open("signals/signals.json") as f:
    signals = json.load(f)
    print(f"Found {len(signals)} signals")
    for s in signals:
        print(f"  - {s.get('signal_id')}: {s.get('direction')}")
```

**Check 4: Bot Logs**
Look for these log messages:
- `External signal handler initialized`
- `Processing validated signal`
- `Signal rejected: <reason>`

### Duplicate Signals Not Detected

**Check:** Persistent deduplication storage exists:
```bash
ls -la signals/.processed_signals.json
```

This file stores processed signal hashes. If missing, deduplication starts fresh.

### Symbol Mapping Failures

**Check:** Symbol is in mapping:
```python
from trading.external_signal_handler import SymbolMapper
mapper = SymbolMapper()
print(mapper.map_symbol("YOUR_SYMBOL"))  # Should not be None
```

If `None`, add custom mapping:
```bash
EXTERNAL_SYMBOL_MAP=YOUR_SYMBOL:deriv_symbol
```

### Stale Signal Errors

Signals older than `EXTERNAL_SIGNAL_MAX_AGE_SECONDS` are rejected. Increase this value if needed:
```bash
EXTERNAL_SIGNAL_MAX_AGE_SECONDS=120  # 2 minutes
```

### Confidence Threshold Filtering

If signals are being rejected for low confidence:
```bash
# Lower the threshold
EXTERNAL_SIGNAL_CONFIDENCE_THRESHOLD=0.5

# Or set to 0 to accept all
EXTERNAL_SIGNAL_CONFIDENCE_THRESHOLD=0.0
```

### MT5 EA Not Writing Signals

**Check 1:** EA is attached to chart and running (smiley face icon)

**Check 2:** Algo trading is enabled (Alt+A)

**Check 3:** Check MT5 Experts tab for errors

**Check 4:** Verify indicator is providing signals (check indicator buffers)

**Check 5:** Check MT5 Journal for write errors

### JSON Corruption

If the signal file becomes corrupted:
1. Delete `signals/signals.json`
2. Bot will recreate it on next start
3. Deduplication storage preserves processed signal history

## Risk Management

External signals inherit all existing risk controls:

- **Minimum Balance**: Trading stops if balance < `MIN_BALANCE`
- **Stake Sizing**: Based on `TRADE_RISK_PERCENT` and `MAX_STAKE`
- **Consecutive Losses**: Stops after `MAX_CONSECUTIVE_LOSSES`
- **Martingale**: Controlled by `MARTINGALE_MODE`

## Monitoring

### Handler Status

The handler provides status information:

```python
# In Python, access handler status
status = signal_handler.get_status()
print(f"Running: {status['running']}")
print(f"Signals received: {status['stats']['signals_received']}")
print(f"Signals validated: {status['stats']['signals_validated']}")
print(f"Trades executed: {status['stats']['trades_executed']}")
```

### Log Messages

Key log messages to monitor:

```
INFO: External signal handler initialized | file=signals/signals.json | max_age=60s
INFO: Starting external signal monitoring thread
INFO: Processing validated signal | id=XXX | source=mt5_ea | symbol=EURUSD->frxEURUSD
INFO: Executing external trade | signal_id=XXX | direction=CALL | stake=$10.00
INFO: Trade executed successfully | contract_id=XXX | signal_id=XXX
INFO: [EXTERNAL] Contract XXX | Signal=CALL | Entry=1.0850 | P/L=5.00
```

## Known Limitations

1. **File-Based Latency**: Small delay between signal generation and execution (typically < 1 second)

2. **Single Writer**: Only one MT5 EA should write to the signal file to avoid conflicts

3. **Same Machine Preferred**: Best performance when MT5 and Python run on same machine

4. **No Acknowledgment**: MT5 EA doesn't receive confirmation of trade execution

## Future Improvements

- [ ] ZeroMQ/Socket-based communication for lower latency
- [ ] Bidirectional communication (acknowledgments back to MT5)
- [ ] Multiple signal source support with priority routing
- [ ] Signal performance tracking by source/strategy
- [ ] Web dashboard for monitoring signal flow

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review bot logs for error messages
3. Verify MT5 EA is running and writing signals
4. Ensure signal file format matches schema
