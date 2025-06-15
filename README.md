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

```bash
git clone https://github.com/yourname/binary-options-bot.git
cd binary-options-bot

# Recommended: Python 3.10+ venv
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> **Tip:** GPU users – install `tensorflow[gpu]` & make sure CUDA 11.8 is on PATH.

---

## 🔑 Environment Variables (`.env`)

| Key                  | Description                                |
| -------------------- | ------------------------------------------ |
| `API_TOKEN`          | Deriv API token                            |
| `APP_ID`             | Deriv App‑ID                               |
| `ASSET`              | Symbol e.g. `R_100`                        |
| `TIMEFRAME`          | Candle timeframe (minutes)                 |
| `EXPIRY`             | Binary expiry (minutes)                    |
| `MIN_BALANCE`        | Abort if equity below                      |
| `MODEL_PATH_LSTM`    | `models/lstm_model.keras`                  |
| `MODEL_PATH_XGB`     | `models/xgb_model.json`                    |
| `MODEL_PATH_RL`      | `models/ppo_agent.zip`                     |
| `TRADE_HISTORY_PATH` | CSV log (default `logs/trade_history.csv`) |
| `SIGNAL_THRESHOLD`   | 0.6 by default                             |
| `MAX_STAKE`          | hard cap per trade                         |

---

## 🚦 Usage

### 1. Train models

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

1. Fetch 2 000 candles via Deriv WebSocket
2. Compute indicators (pandas‑ta)
3. Rule strategies + LSTM/XGB → directions  
   _Optionally_ vote for ≥3 agreement
4. RL agent sizes stake
5. Execute via `proposal → buy`
6. On settle → log to `trade_history.csv` + update `confidence_tracker.json`

---

## Confidence Tracker

A dynamic system that chooses which strategy to use based on their performance over time.
| Field | Meaning |
| -------------- | ----------------------------- |
| `confidence` | Decay‑weighted win‑rate (0‑1) |
| `last_updated` | ISO timestamp |

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

## 🛡️ Risk Controls

- Config stop‑trading if balance < `MIN_BALANCE`
- RL action=0 forces skip
- Stake is min(balance × 1 %, `MAX_STAKE`)
- Confidence threshold gates low‑prob trades

---

## 🤝 Contributing

PRs welcome! Please:

1. Open issue describing change
2. Follow `black` + `isort`

---

## 📜 License

MIT – see `LICENSE`.

---

## 💬 Contact

Raise an issue or ping me on [Twitter](https://twitter.com/yourhandle).
