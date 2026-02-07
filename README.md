# Binary Options Bot 🚀📈

A modular, research‑grade trading framework for **binary options** on Deriv (and other brokers (more of these later)).  
It blends _rule‑based ensembles_, _deep‑learning price forecasting_, _reinforcement‑learning_ stake sizing
and a **self‑improving confidence tracker** – all wrapped in an async Python engine.

---

## 🌐 High‑Level Architecture

```
binary_options_bot/
│
├── main.py                         # Entrypoint: runs the trading loop
├── .env                            # Environment variables (API keys, config)
├── requirements.txt                # Python dependencies
├── README.md                       # You're reading it :)
│
├── broker/                         # Broker interface (Deriv API)
│   ├── __init__.py
│   └── deriv.py
│
├── data/                           # Dataset & metadata
│   ├── historical_candles.csv
│   ├── historical_features.csv
│   ├── rl_training_data.npy
│   └── strategy_confidence.json   # Dynamic strategy confidence scores
│
├── logs/
│   └── trade_history.csv           # Execution history log
│
├── models/                         # Trained models (ML + RL)
│   ├── lstm_model.keras
│   ├── lstm_best.keras
│   ├── ppo_agent.zip
│   ├── xgb_model.json
│   └── *.png / .txt               # Visualizations + metrics
│
├── ml/                             # ML training pipeline
│   ├── __init__.py
│   ├── training_pipeline.py
│   ├── train_rl_agent.py
│   └── __main__.py
│
├── reinforcements/                # RL agent definition (e.g. PPO)
│   ├── __init__.py
│   ├── agent.py
│   └── environment.py
│
├── notifier/                      # Notification service (Telegram, etc.)
│   ├── __init__.py
│   └── telegram.py
│
├── trading/                        # Core trading logic
│   ├── __init__.py
│   ├── strategy.py                 # TA + AI signal generation
│   ├── executor.py                 # Trade execution logic
│   ├── backtest.py                 # Single-strategy backtesting
│   ├── backtest_voting.py         # Voting-based ensemble backtesting
│   ├── confidence.py              # Confidence tracker (learns over time)
│   ├── ml_strategy_logic.md       # Markdown notes for ML strategy design
│   └── technical_analysis_strategies.py # Extracted from literature (books)
│
├── backtest_charts/               # Visual results
│   ├── [strategy_name]/...

```

---

## ⚙️ Installation

> **Prerequisites**
>
> - Git + Conda + Python 3.10 (only for the local route).
> - Docker 23 + (This is the preferred option).
> - A valid `env` file with your API keys & runtime settings.


### ▶️ Option A – Local (Python venv) Or Conda

```bash
# 1. Grab the code
git clone https://github.com/yourname/binary-options-bot.git
cd binary-options-bot

# 2. Create & activate a Python 3.10 virtual environment
python -m venv .venv      # macOS/Linux
# py -3 -m venv .venv     # Windows
source .venv/bin/activate # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
conda install -c conda-forge ta-lib  # Optional: for technical analysis of some strategies
```

Run the bot:

```bash
python main.py
```

---

### ▶️ Option B – Docker (no docker-compose)

1. **Build the image**

   ```bash
   docker build -t binary-options-bot .
   ```

2. **Run the container**

   ```bash
   docker run -d --env-file env binary-options-bot
   ```

3. **Known Issue with logs in docker**

   ```bash
   docker logs -f <container_id>
   # This doesn't show anything and I'm open to suggestions on how to fix it.
   ```

   That’s it—choose the route that best suits your workflow. Happy trading! 🚀

---

## 🔑 Environment Variables (`env`)

**Get your APPID and API keys from [Deriv API](https://api.deriv.com/dashboard/)** and set them in the `env` file in the root directory.

```bash
# Copy .env.example to env (note that there's no '.' infront of env ) and fill in your APP ID, API token from Deriv
```

Configure the bot behavior by editing the `env` file:

| Key                | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `APP_ID`           | Your Deriv App ID                                                      |
| `API_TOKEN`        | Your Deriv API token                                                   |
| `ASSET`            | Trading symbol (e.g., `R_100`)                                         |
| `EXPIRY`           | Trade expiry in minutes (e.g., `5`)                                    |
| `STAKE`            | Default stake amount per trade                                         |
| `TIMEFRAME`        | Candle timeframe in minutes (e.g., `1`)                                |
| `TELEGRAM_TOKEN`   | Telegram bot token for trade notifications                             |
| `TELEGRAM_CHAT_ID` | Chat ID to receive Telegram alerts                                     |
| `BACKTEST`         | Whether to run backtesting instead of live trading (`true` or `false`) |

### 🧠 Model Configuration

| Key                | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `MODEL_TYPE`       | Type of model to use: `lstm`, `xgboost`, `hybrid`, or `rule` |
| `MODEL_PATH_LSTM`  | Path to LSTM model file (`.h5`)                              |
| `MODEL_PATH_XGB`   | Path to XGBoost model file (`.json`)                         |
| `MODEL_PATH_RL`    | Path to Reinforcement Learning model (`.zip`)                |
| `SIGNAL_THRESHOLD` | Confidence threshold to trigger trades (`0.0`–`1.0`)         |
| `SEQUENCE_LENGTH`  | Number of timesteps to use for sequence models (e.g., `100`) |

### 💵 Risk Management

| Key                      | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `MAX_CONSECUTIVE_LOSSES` | Stop trading after this many losses in a row               |
| `TRADE_RISK_PERCENT`     | Risk per trade as a percentage of current balance          |
| `MIN_BALANCE`            | Halt trading if balance drops below this threshold         |
| `MARTINGALE_MODE`        | Position sizing: `off`, `martingale`, or `anti-martingale` |

### 📊 Backtesting

| Key                    | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `HISTORICAL_DATA_PATH` | Path to CSV with historical candle data              |
| `SAVE_TRADE_HISTORY`   | Whether to save trades to a file (`true` or `false`) |
| `TRADE_HISTORY_PATH`   | File path where trade history is stored              |

---

## 🚦 Usage

### 1. Train models (Not in use.)

**This has been commented out in the code and the requirements as well have been commented out**
If you want to use models, uncomment the code in `requirements.txt` and `trading/strategy.py`:
Just a heads up:
**_Predicting the Financial markets with AI/ML is dead difficult😭_**

```bash
python ml/training_pipeline.py          # LSTM + XGB
python ml/train_rl_agent.py             # PPO stake‑sizer
```

### 2. Back‑test rule/AI/ensemble

```bash
python trading/backtest.py              # single‑strategy loop
python trading/backtest_ensemble.py     # ≥3‑agreement voting system
```

Charts saved to **`backtest_charts/`** and summary CSV.

### 3. Go live (paper‑trading or real)

```bash
python main.py
```

Live loop:

## 🔁 Trading Logic Flow

### 📊 Rule-Based Path

1. **Fetch Candle Data**
   → Stream 2,000 candles via Deriv WebSocket (real-time)

2. **Apply Technical Indicators**
   → Compute momentum, trend, volatility indicators using `pandas-ta`

3. **Generate Rule-Based Signals**
   → Apply preconfigured rule-based strategies (e.g., RSI, EMA crossovers)

4. **Optional Voting Logic (Currently in use)**
   → If 3 or more rule strategies agree on the same direction(CALL or PUT), proceed

5. **Stake Calculation**
   → Use fixed or 1% of balance

6. **Trade Execution**
   → Send `proposal → buy` via Deriv API

7. **On Settlement**
   → Log trade result to `logs/trade_history.csv`
   → Update `logs/confidence_tracker.json` (to monitor signal strength over time)

---

### 🤖 AI-Powered Path (Not in Use)

1. **Fetch Candle Data**
   → Stream 2,000 candles via Deriv WebSocket (real-time)

2. **Prepare Model Input**
   → Use historical candles and indicators to generate input features

3. **Run ML Models**
   → LSTM and/or XGBoost infer market direction (CALL / PUT)

4. **RL Agent Sizing**
   → PPO-based agent determines stake size and adjusts dynamically

5. **Trade Execution**
   → Send `proposal → buy` via Deriv API

6. **On Settlement**
   → Log to:

   - `logs/trade_history.csv`
   - `logs/confidence_tracker.json` (to monitor signal strength over time)

---

### 📡 MT4/MT5 External Signal Integration

The bot supports receiving signals from MT4/MT5 platforms through a file-based bridge:

1. **MT5 Indicator** generates signals based on technical analysis
2. **MT5 Expert Advisor** reads the indicator and writes signals to `signals/signals.json`
3. **Python Signal Handler** monitors the file and executes trades based on received signals
4. **Existing Execution Logic** processes the external signals using the same risk management

#### Signal Format
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

To enable external signals, set `ENABLE_EXTERNAL_SIGNALS=true` in your env file.

## Confidence Tracker

A dynamic system that chooses which strategy to use based on it's performance over time.
| Field | Meaning |
| -------------- | ----------------------------- |
| `confidence` | Decay‑weighted win‑rate (0‑1) |
| `last_updated` | ISO timestamp |

Eg:

```json
{
  "sma_rsi": {
    "confidence": 0.2458,
    "last_updated": "2025-06-15T21:38:02.432198"
  },
  "macd_cross": {
    "confidence": 0.45,
    "last_updated": "2025-06-15T21:14:00.295129"
  },
  "bollinger_bands": {
    "confidence": 0.6548,
    "last_updated": "2025-06-15T18:46:02.350494"
  }
}
-> This example means that if all these 3 strategies agree on a direction(say PUT), the bot will execute a trade with the bollinger bands strategy because it has a higher confidence.

```

Update rule:

```python
new = old * decay + result * (1‑decay)   # decay=0.8
```

Use `get_confidence(strat)` inside `strategy.py` to bias voting toward high‑performers.

---

## 🧠 Strategy Catalogue (rule‑based)

- `sma_rsi`
- `macd_cross`
- `bollinger_bands`
- `adx_trend`
- `ichimoku_base_conversion`
- `breakout`
- `trend_follow`
- `stoch_rsi`
- `heikin_ashi`
- `candle_reversal`
- `atr_breakout`
- `psar`
- `vwap`
- `ema_ribbon`
- `vol_spike`
- `golden_death_cross`
- `price_channel`

> Add your own in `strategy.py` – just return `(direction, confidence)`.

---

## 📈 Logging & Analytics

| File                     | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `logs/trade_history.csv` | Per‑trade journal (timestamp, strategy, P/L) |
| `backtest_charts/*.png`  | Equity curves per strategy                   |
| `backtest_summary.csv`   | Final balances & win‑rates                   |

---

## 📊 Backtesting

### Quick Performance Summary (Account Balance was initially $1000)

**This is a summary of the backtest results for each strategy across different expiry timeframes (1min to 5min). The values represent the final account balance after running the backtest.**

#### Screenshots of backtest results are saved in `backtest_charts/`.

| Duration                             | 1min    | 2min    | 3min    | 4min    | 5min    |
| ------------------------------------ | ------- | ------- | ------- | ------- | ------- |
| **Strategy**                         |
| adx_trend                            | 729.21  | 1150.79 | 1107.85 | 1365.52 | 1195.38 |
| atr_breakout                         | 748.99  | 707.47  | 655.67  | 808.17  | 872.02  |
| bollinger_bands                      | 759.88  | 582.32  | 446.25  | 529.52  | 429.60  |
| bollinger_breakout                   | 745.57  | 972.91  | 1269.57 | 1069.92 | 1318.77 |
| breakout                             | 676.60  | 899.86  | 989.59  | 1130.44 | 1067.77 |
| candle_reversal                      | 980.10  | 980.10  | 980.10  | 998.91  | 998.91  |
| doon_volatility_contraction          | 1000.00 | 1000.00 | 1000.00 | 1000.00 | 1000.00 |
| ena_ribbon                           | 993.94  | 1072.47 | 1072.47 | 1135.41 | 1052.28 |
| golden_death_cross                   | 977.13  | 1014.99 | 1054.33 | 1074.56 | 1054.33 |
| heikh_ashi                           | 81.84   | 49.93   | 80.30   | 104.79  | 126.73  |
| ichimoku_base_conversion             | 301.48  | 440.94  | 961.34  | 1488.56 | 1905.87 |
| macd_cross                           | 104.79  | 117.45  | 256.06  | 436.03  | 162.26  |
| price_channel                        | 1000.00 | 1000.00 | 1000.00 | 1000.00 | 1000.00 |
| psar                                 | 329.44  | 355.46  | 438.14  | 691.44  | 390.91  |
| sentinent_volatility_stochastic_fade | 926.86  | 981.25  | 981.25  | 981.25  | 909.40  |
| sna_rsi                              | 944.06  | 858.46  | 874.93  | 810.87  | 751.50  |
| sna_rsi_macd_combo                   | 1009.00 | 1009.00 | 1009.00 | 1009.00 | 990.00  |
| stoch_rsi                            | 638.39  | 462.10  | 638.39  | 614.57  | 650.64  |
| stoch_rsi_reversal                   | 712.51  | 525.65  | 754.33  | 783.56  | 783.56  |
| trend_follow                         | 688.78  | 1218.32 | 1172.87 | 1241.71 | 1172.87 |
| vol_spike                            | 1000.00 | 1000.00 | 1000.00 | 1000.00 | 1000.00 |
| volatility_ladder_strategy           | 984.85  | 1003.75 | 1003.75 | 948.11  | 984.85  |
| vwap                                 | 1000.00 | 1000.00 | 1000.00 | 1000.00 | 1000.00 |

## 🤝 Contributing

PRs welcome! Please:

**I am particularly looking for help on the notifications with telegram and the docker logs showing**

Apart from that, all suggestions are welcome.

## 📜 License

MIT – see [`LICENSE`](LICENSE.md).

---

## 💬 Contact

Raise an issue, give a suggestion or ping me on [LinkedIn](https://www.linkedin.com/in/alexis-selorm/).

# 🗣 Final Wise Words

- "The market is a device for transferring money from the impatient to the patient." – Warren Buffett
- "In trading and investing, it's not about how much you make but rather how much you don't lose." – Bernard Baruch
- "The four most dangerous words in investing are: 'This time it's different.'" – Sir John Templeton
- "Risk comes from not knowing what you're doing." – Warren Buffett

### 🧠 From Me

> _“This bot isn’t magic—it’s math, code, and a bit of stubborn curiosity. Whether it prints profits or teaches hard lessons, it’s mine. Fork it. Improve it. Or just watch it tick.”_

### 🤖 From the Bot

> _“I don’t sleep. I don’t panic. I don’t revenge trade. I process signals and act. If you feed me data and logic, I’ll do the rest. Let’s see if your edge is real.”_

### ✨ Inspiration for the nerds like me

> _“Built for the curious. Shared for the bold. Use this bot as a blueprint, a launchpad, or a challenge. Whatever you do—trade smarter.”_

---
