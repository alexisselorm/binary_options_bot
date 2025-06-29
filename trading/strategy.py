# trading/strategy.py
"""Hybrid signal engine (AI + pluggable rule catalogue)
----------------------------------------------------------------
• **AI stack:**  LSTM  →  XGBoost  (+ optional PPO‑RL stake sizing)
• **Rule stack:**  multiple TA strategies selectable at call‑time.

`generate_signals()` signature (unchanged)
    df, balance, *, use_ai=True, strategy="sma_rsi"
returns `(signal | None, stake | None, confidence | None)`
"""
from __future__ import annotations
from trading.tecnhical_analysis_strategies import *

import os
import json
import requests
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re

# import xgboost as xgb
# from tensorflow.keras.models import load_model

# from reinforcements.agent import RLAgent
from utils.config import Config
# ════════════════════════════════════════════════════════════════
# One‑time configuration / model loads
# ════════════════════════════════════════════════════════════════
_cfg = Config()

SEQ_LEN: int = _cfg.sequence_length   # e.g. 200 timesteps
THRESH:  float = _cfg.signal_threshold

# ════════════════════════════════════════════════════════════════
# Indicator helper – used by **all** rule strategies and AI feature set
# ════════════════════════════════════════════════════════════════

# Import everything from technical_analysis_strategies


def collect_rule_signals(
    df: pd.DataFrame,
    balance: float,
    min_agree: int = 2,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run *all* rule strategies and return (direction, details)
      • direction  : "CALL" | "PUT" | None
      • details    : string listing which strats agreed

    Trades only when `min_agree` or more strategies give *identical* direction.
    """
    df = prepare_df_for_ta(df)
    df = add_indicators(df)
    votes: Dict[str, List[str]] = {"CALL": [], "PUT": []}

    for strat_name, rule_fn in RULE_BASED_STRATEGIES.items():
        direction, _ = rule_fn(df)            # stake/conf not needed here
        if direction in ("CALL", "PUT"):
            votes[direction].append(strat_name)

    for side, agreeing in votes.items():
        if len(agreeing) >= min_agree:
            return side, ", ".join(agreeing)

    return None, ""  # no consensus


# ════════════════════════════════════════════════════════════════
# Main generator
# ════════════════════════════════════════════════════════════════


def generate_signals(
    df: pd.DataFrame,
    balance: float,
    *,
    use_ai: bool = False,
    strategy: str,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (signal, stake, confidence).  If no trade → (None, None, None)"""

    print(f"Use AI? {use_ai}")
    if not use_ai and strategy not in RULE_BASED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {RULE_STRATEGY_NAMES}")

    df = prepare_df_for_ta(df)
    df = add_indicators(df)
    if len(df) < SEQ_LEN:
        print(f"Len of df: {len(df)}")
        print(f"Len of SEQ: {SEQ_LEN}")
        return None, None, None

    # ──────────────────────────────────────────────────────────
    # Rule‑based branch
    # ──────────────────────────────────────────────────────────
    if not use_ai:
        sig, conf = RULE_BASED_STRATEGIES[strategy](df)
        if sig is None:
            return None, None, None
        stake = min(balance * 0.01, _cfg.max_stake)
        return sig, stake, conf

    print("Skipped everything, now using llm")
    # ──────────────────────────────────────────────────────────
    # LLM-assisted AI branch
    # ──────────────────────────────────────────────────────────
    latest = df.tail(SEQ_LEN).copy()
    latest.index = pd.to_datetime(latest['epoch'], unit='s')
    latest = latest.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low", "close": "Close",
        "volume": "Volume"
    })

    # Plot to buffer
    img_buf = BytesIO()
    mpf.plot(
        latest,
        type='candle',
        style='charles',
        volume=True,
        mav=(3, 6),
        show_nontrading=True,
        savefig=dict(fname=img_buf, dpi=150, bbox_inches='tight')
    )
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
    # Save plot to disk for later review
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
    with open(plot_path, "wb") as f:
        f.write(base64.b64decode(img_b64))
    print(f"Plot saved to {plot_path}")
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {
                "role": "user",
                "content": "This is a 1-minute candlestick chart for R_100. Based on this chart, provide:\n- Trading Signal: CALL, PUT, or NONE\n- Confidence (0–100%)\n- Reasoning (short).",
                "images": [img_b64]
            }
        ]
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_cfg.llm_api_token}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost.5534",
                "X-Title": "BinaryOptionsBot"
            },
            data=json.dumps(payload),
            timeout=15
        )
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"].lower()
        print(f"Content 📖: {content}")
        if "call" in content:
            signal = "CALL"
        elif "put" in content:
            signal = "PUT"
        else:
            return None, None, None

        import re
        conf_match = re.search(r"(\d{1,3}(\.\d+)?)%", content)
        confidence = float(conf_match.group(1)) if conf_match else 50.0

        if confidence < THRESH:
            print(f"❌ LLM Conf {confidence:.2f} < {THRESH}. Skip")
            return None, None, None

        stake = min(balance * 0.01, _cfg.max_stake)
        return signal, stake, confidence

    except Exception as e:
        print(f"❌ LLM error: {e}")
        return None, None, None

 # ──────────────────────────────────────────────────────────
    # AI branch
    # ──────────────────────────────────────────────────────────
    ai = """
    AI signal generation:
      • LSTM predicts next close price
      • XGBoost uses LSTM output + technical features to predict CALL/PUT
      • RL agent (optional) adjusts stake size based on confidence
    
    JUST A SIDE NOTE: Predicting financial markets is extremely difficult. 😭

    # — LSTM —
    _lstm = load_model(_cfg.model_path_lstm)
    _lstm.compile()

    # — XGBoost —
    _xgb = xgb.XGBClassifier()
    _xgb.load_model(_cfg.model_path_xgb)

    # — RL (optional) —
    _rl: Optional[RLAgent] = (
        RLAgent(model_path=_cfg.model_path_rl)
        if _cfg.model_path_rl and os.path.exists(_cfg.model_path_rl)
        else None
    )

    "❌ Financial Prediction with AI/ML is HARD"
    last = df.iloc[-1]
    feature_cols = get_feature_columns(df)
    tech_features = [last[c] for c in feature_cols]

    seq = df[["open", "high", "low", "close"]
             ].values[-SEQ_LEN:].astype(np.float32)
    seq = seq.reshape(1, SEQ_LEN, 4)
    lstm_pred = float(_lstm.predict(seq, verbose=0)[0][0])
    tech_features.append(lstm_pred)

    prob_call = float(_xgb.predict_proba([tech_features])[0][1])
    prob_put = 1.0 - prob_call
    confidence = max(prob_call, prob_put)
    signal = "CALL" if prob_call > prob_put else "PUT"

    print(f"📊 AI signal={signal}  conf={confidence:.2f}")

    if confidence < THRESH:
        print(f"❌ Conf {confidence:.2f} < {THRESH}. Skip")
        return None, None, None

    # No RL → fixed 1 % stake
    if _rl is None:
        return signal, min(balance * 0.01, _cfg.max_stake), confidence

    # RL stake sizing
    rl_action, stake_mult = _rl.predict(
        np.array([confidence, lstm_pred], dtype=np.float32))
    if rl_action == 0:
        print("🧠 RL skip")
        return None, None, None
    stake_usd = min(balance * max(stake_mult, 0.01), _cfg.max_stake)
    print(f"🧠 RL proceed stake={stake_usd:.2f} (mult={stake_mult:.2f})")
    return signal, stake_usd, confidence
    """
