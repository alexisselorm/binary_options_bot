import pandas as pd
import numpy as np
import pandas_ta as ta
from functools import wraps
from trading.confidence import get_confidence

from typing import Tuple, Optional, List, Dict, Callable


def bollinger_breakout(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Source: Trading Binary Options – Abe Cofnas (Chapter 6, p. 100)
    Rule: Price closes outside Bollinger Band → trade in direction of breakout
    Timeframe: 15-min or hourly
    """
    df = df.copy()
    df.ta.bbands(length=20, std=2, append=True)

    if len(df) < 2:
        return None, None

    last_close = df["close"].iloc[-1]
    bbu = df["BBU_20_2.0"].iloc[-1]
    bbl = df["BBL_20_2.0"].iloc[-1]

    if last_close > bbu:
        return "CALL", 0.02  # Stake 2%
    elif last_close < bbl:
        return "PUT", 0.02

    return None, None


def stoch_rsi_reversal(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Source: Trading Binary Options – Abe Cofnas (Chapter 6, p. 100)
    Rule: Stochastic RSI < 20 AND RSI < 30 → CALL | >80 & >70 → PUT
    Adds ADX filter to reduce false signals [⚠️ Inference]
    """
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.adx(length=14, append=True)

    if len(df) < 2:
        return None, None

    rsi = df["RSI_14"].iloc[-1]
    stoch_k = df["STOCHk_14_3_3"].iloc[-1]
    adx = df["ADX_14"].iloc[-1]

    if pd.isna(rsi) or pd.isna(stoch_k):
        return None, None

    if adx < 25:
        return None, None

    if stoch_k < 20 and rsi < 30:
        return "CALL", 0.015  # Stake 1.5%
    elif stoch_k > 80 and rsi > 70:
        return "PUT", 0.015

    return None, None


def doom_volatility_contraction(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Source: Binary Options – Alex Nekritin (Chapter 8, p. 108–116)
    Rule: Narrowing bands + price near key level → Buy DOOM option
    Timeframe: Daily
    """
    df = df.copy()
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    if len(df) < 2:
        return None, None

    band_width = df["BBU_20_2.0"] - df["BBL_20_2.0"]
    is_narrowing = band_width.pct_change().iloc[-1] < 0
    close = df["close"].iloc[-1]
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    if not is_narrowing:
        return None, None

    if close < support:
        return "CALL", 0.01  # Stake 1%
    elif close > resistance:
        return "PUT", 0.01

    return None, None


def volatility_ladder_strategy(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Source: Binary Options – Alex Nekritin + Trading Binary Options – Abe Cofnas
    Rule: BB contraction + price near S/R → ladder multiple entries
    """
    df = df.copy()
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    if len(df) < 2:
        return None, None

    band_width = df["BBU_20_2.0"] - df["BBL_20_2.0"]
    is_narrowing = band_width.pct_change().iloc[-1] < 0
    close = df["close"].iloc[-1]
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    if not is_narrowing:
        return None, None

    if close <= support:
        return "CALL", 0.02  # Max 2% total
    elif close >= resistance:
        return "PUT", 0.02

    return None, None


def sentiment_volatility_stochastic_fade(
    df: pd.DataFrame,
    sentiment_signal: str = "neutral"
) -> Tuple[Optional[str], Optional[float]]:
    """
    Source: Trading Binary Options – Abe Cofnas + Binary Options – Alex Nekritin
    Rule: Fade oversold/overbought with sentiment confirmation
    """
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)

    if len(df) < 2:
        return None, None

    stoch_k = df["STOCHk_14_3_3"].iloc[-1]
    rsi = df["RSI_14"].iloc[-1]

    band_width = df["BBU_20_2.0"] - df["BBL_20_2.0"]
    is_contracting = band_width.pct_change().iloc[-1] < 0

    if not is_contracting:
        return None, None

    if stoch_k < 20 and rsi < 30:
        if sentiment_signal == "bearish":
            return None, None
        return "CALL", 0.015  # Stake 1.5%

    elif stoch_k > 80 and rsi > 70:
        if sentiment_signal == "bullish":
            return None, None
        return "PUT", 0.015

    return None, None


WARNING_SUPPRESSED_STRATEGIES = {
    "_heikin_ashi",
    "_vwap",
    "_ichimoku_base_conversion"
}


def safe_strategy(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if result is None:
                return None, None
            if not isinstance(result, tuple) or len(result) != 2:
                return None, None
            return result
        except Exception as e:
            print(f"💥 Strategy error [{fn.__name__}]: {e}")
            return None, None
    return wrapper


def prepare_df_for_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index + sorted for proper TA handling"""
    df = df.copy()
    df.index = pd.to_datetime(df["epoch"], unit="s")
    df = df.sort_index()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the indicator set required by ALL rule strategies
    (old + the new ones you added).
    """
    df = df.copy()

    # ── basic OHLC sanity ─────────────────────────────────────────
    # Deriv “candle” payloads sometimes omit volume → make a dummy
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # ── trend / averages ─────────────────────────────────────────
    ema_lengths = [8, 13, 21, 34, 55]          # extra EMAs for ribbon
    for ln in ema_lengths:
        df.ta.ema(length=ln, append=True)

    sma_lengths = [50, 200]                    # golden/death cross
    for ln in sma_lengths:
        df.ta.sma(length=ln, append=True)

    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.ichimoku(append=True)

    # ── momentum ────────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(length=14, append=True)        # STOCHk_14_3_3

    # ── volatility / bands ──────────────────────────────────────
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.atr(length=14, append=True)          # ATRr_14 used by breakout

    df.dropna(inplace=True)
    return df


# columns to feed to XGB (skip raw price cols)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"open", "high", "low", "close", "epoch"}
    return [c for c in df.columns if c not in exclude]


# ════════════════════════════════════════════════════════════════
# Rule‑based strategies
# ════════════════════════════════════════════════════════════════
RuleFn = Callable[[pd.DataFrame], Tuple[Optional[str], Optional[float]]]


@safe_strategy
def sma_rsi(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]
    fast, slow, rsi = last["SMA_50"], last["SMA_200"], last["RSI_14"]
    if fast > slow and rsi < 30:
        return "CALL", get_confidence("sma_rsi")
    if fast < slow and rsi > 70:
        return "PUT", get_confidence("sma_rsi")
    return None, None


@safe_strategy
def macd_cross(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]
    macd, signal = last["MACD_12_26_9"], last["MACDs_12_26_9"]
    if macd > signal:
        return "CALL", get_confidence("macd_cross")
    if macd < signal:
        return "PUT", get_confidence("macd_cross")
    return None, None


@safe_strategy
def sma_rsi_macd_combo(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    """
    Combines the logic of `sma_rsi` and `macd_cross` strategies.
    Only returns a signal if both strategies agree.
    """
    last = df.iloc[-1]

    # Get values
    fast = last["SMA_50"]
    slow = last["SMA_200"]
    rsi = last["RSI_14"]
    macd = last["MACD_12_26_9"]
    signal = last["MACDs_12_26_9"]

    # Check for CALL condition
    if fast > slow and rsi < 30 and macd > signal:
        confidence = get_confidence("sma_rsi_macd_combo")
        return "CALL", confidence

    # Check for PUT condition
    elif fast < slow and rsi > 70 and macd < signal:
        confidence = get_confidence("sma_rsi_macd_combo")
        return "PUT", confidence

    # If no agreement
    return None, None


@safe_strategy
def bollinger_bands(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]
    if last["close"] < last["BBL_20_2.0"]:
        return "CALL", get_confidence("bollinger_bands")
    if last["close"] > last["BBU_20_2.0"]:
        return "PUT", get_confidence("bollinger_bands")
    return None, None


@safe_strategy
def adx_trend(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]
    if last["ADX_14"] > 25:
        if last["DMP_14"] > last["DMN_14"]:
            return "CALL", get_confidence("adx_trend")
        if last["DMP_14"] < last["DMN_14"]:
            return "PUT", get_confidence("adx_trend")
    return None, None


@safe_strategy
def ichimoku_base_conversion(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]

    # Conversion (Tenkan-sen) vs. Base (Kijun-sen)
    conv = last["ITS_9"]   # Tenkan-sen
    base = last["IKS_26"]  # Kijun-sen  ← fixed column name

    if conv > base:
        return "CALL", get_confidence("ichimoku_base_conversion")
    if conv < base:
        return "PUT", get_confidence("ichimoku_base_conversion")
    return None, None


@safe_strategy
def breakout(df: pd.DataFrame, look: int = 20) -> Tuple[Optional[str], Optional[float]]:
    if len(df) < look + 1:
        return None, None
    high_lvl = df["high"].rolling(look).max().iloc[-2]
    low_lvl = df["low"].rolling(look).min().iloc[-2]
    last = df.iloc[-1]
    if last["close"] > high_lvl:
        return "CALL", get_confidence("breakout")
    if last["close"] < low_lvl:
        return "PUT", get_confidence("breakout")
    return None, None


@safe_strategy
def trend_follow(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    last = df.iloc[-1]
    ema_f = last["EMA_8"]
    ema_s = last["EMA_21"]
    adx = last["ADX_14"]
    if ema_f > ema_s and adx > 25:
        return "CALL", get_confidence("trend_follow")
    if ema_f < ema_s and adx > 25:
        return "PUT", get_confidence("trend_follow")
    return None, None


@safe_strategy
def stoch_rsi(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    try:
        if len(df) < 2:
            return None, None

        required_cols = ["STOCHk_14_3_3", "RSI_14"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"⚠️ Skipping '_stoch_rsi': missing indicators {missing}")
            return None, None

        last = df.iloc[-1]
        stoch_k = last["STOCHk_14_3_3"]
        rsi = last["RSI_14"]

        if pd.isna(stoch_k) or pd.isna(rsi):
            return None, None

        if stoch_k < 20 and rsi < 30:
            return "CALL", get_confidence("_stoch_rsi")
        if stoch_k > 80 and rsi > 70:
            return "PUT", get_confidence("_stoch_rsi")

    except Exception as e:
        print(f"❌ Error in '_stoch_rsi': {e}")

    return None, None


@safe_strategy
def heikin_ashi(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    try:
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df["epoch"], unit="s")
            df = df.sort_index()

        # Generate HA candles
        df.ta.ha(append=True)

        # Recalculate HA Open safely
        for i in range(1, len(df)):
            prev_open = df.loc[df.index[i - 1], "HA_open"]
            prev_close = df.loc[df.index[i - 1], "HA_close"]
            df.loc[df.index[i], "HA_open"] = 0.5 * (prev_open + prev_close)

        ha_close_col = "HA_close"
        ha_open_col = "HA_open"

        if ha_close_col not in df.columns or ha_open_col not in df.columns:
            return None, None

        ha_close = df.loc[df.index[-1], ha_close_col]
        ha_open = df.loc[df.index[-1], ha_open_col]

        if ha_close > ha_open:
            return "CALL", get_confidence("_heikin_ashi")
        elif ha_close < ha_open:
            return "PUT", get_confidence("_heikin_ashi")

    except Exception as e:
        print(f"❌ Error in '_heikin_ashi': {e}")

    return None, None


@safe_strategy
def candle_reversal(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    try:
        df.ta.cdl_pattern(name=["engulfing", "hammer",
                          "shootingstar"], append=True)
        patterns = df.iloc[-1].filter(like="_1").astype(int)
        if any(patterns == 100):  # Bullish reversal
            return "CALL", get_confidence("_candle_reversal")
        if any(patterns == -100):  # Bearish reversal
            return "PUT", get_confidence("_candle_reversal")
    except Exception as e:
        print(f"❌ Error in '_candle_reversal': {e}")
    return None, None


@safe_strategy
def atr_breakout(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    df.ta.atr(length=14, append=True)
    atr = df["ATRr_14"].iloc[-1]
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if curr["close"] > prev["high"] + atr:
        return "CALL", get_confidence("_atr_breakout")
    if curr["close"] < prev["low"] - atr:
        return "PUT", get_confidence("_atr_breakout")
    return None, None


@safe_strategy
def psar(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    df.ta.psar(append=True)
    psar = df["PSARl_0.02_0.2"].iloc[-1] if "PSARl_0.02_0.2" in df.columns else np.nan
    close = df["close"].iloc[-1]

    if not np.isnan(psar):
        if close > psar:
            return "CALL", get_confidence("_psar")
        else:
            return "PUT", get_confidence("_psar")
    return None, None


@safe_strategy
def vwap(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    df = df.copy()
    df.index = pd.to_datetime(df["epoch"], unit="s")
    df = df.sort_index()
    df.ta.vwap(append=True)
    if "VWAP_D" not in df.columns:
        return None, None

    vwap_val = df["VWAP_D"].iloc[-1] if "VWAP_D" in df.columns else np.nan
    close = df["close"].iloc[-1]

    if not np.isnan(vwap_val):
        if close > vwap_val:
            return "CALL", get_confidence("_vwap")
        elif close < vwap_val:
            return "PUT", get_confidence("_vwap")
    return None, None


@safe_strategy
def ema_ribbon(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    emas = [f"EMA_{i}" for i in [8, 13, 21, 34, 55]]
    for e in emas:
        if e not in df.columns:
            return None, None

    vals = df[emas].iloc[-1].values
    if all(vals[i] > vals[i+1] for i in range(len(vals)-1)):  # EMA descending → downtrend
        if df["close"].iloc[-1] < vals[-1]:  # Price below slowest EMA
            return "CALL", get_confidence("_ema_ribbon")
    elif all(vals[i] < vals[i+1] for i in range(len(vals)-1)):  # EMA ascending → uptrend
        if df["close"].iloc[-1] > vals[-1]:
            return "PUT", get_confidence("_ema_ribbon")
    return None, None


@safe_strategy
def vol_spike(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    vol_avg = df["volume"].rolling(20).mean().iloc[-2]
    vol_curr = df["volume"].iloc[-1]
    if vol_curr > 2 * vol_avg:
        if df["close"].iloc[-1] > df["open"].iloc[-1]:
            return "CALL", get_confidence("_vol_spike")
        else:
            return "PUT", get_confidence("_vol_spike")
    return None, None


@safe_strategy
def golden_death_cross(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    sma50 = df["SMA_50"].iloc[-1]
    sma200 = df["SMA_200"].iloc[-1]
    sma50_prev = df["SMA_50"].iloc[-2]
    sma200_prev = df["SMA_200"].iloc[-2]

    # Golden cross
    if sma50_prev < sma200_prev and sma50 > sma200:
        return "CALL", get_confidence("_golden_death_cross")
    # Death cross
    if sma50_prev > sma200_prev and sma50 < sma200:
        return "PUT", get_confidence("_golden_death_cross")
    return None, None


@safe_strategy
def price_channel(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
    lookback = 20
    df["channel_high"] = df["high"].rolling(lookback).max()
    df["channel_low"] = df["low"].rolling(lookback).min()

    last = df.iloc[-1]
    if last["close"] > last["channel_high"]:
        return "CALL", get_confidence("_price_channel")
    if last["close"] < last["channel_low"]:
        return "PUT", get_confidence("_price_channel")
    return None, None


RULE_BASED_STRATEGIES: Dict[str, RuleFn] = {
    "sma_rsi": sma_rsi,
    "macd_cross": macd_cross,
    "sma_rsi_macd_combo": sma_rsi_macd_combo,
    "bollinger_bands": bollinger_bands,
    "adx_trend": adx_trend,
    "ichimoku_base_conversion": ichimoku_base_conversion,
    "breakout": breakout,
    "trend_follow": trend_follow,
    "stoch_rsi": stoch_rsi,
    "heikin_ashi": heikin_ashi,
    "candle_reversal": candle_reversal,
    "atr_breakout": atr_breakout,
    "psar": psar,
    "vwap": vwap,
    "ema_ribbon": ema_ribbon,
    "vol_spike": vol_spike,
    "golden_death_cross": golden_death_cross,
    "price_channel": price_channel,
    "bollinger_breakout": bollinger_breakout,
    "stoch_rsi_reversal": stoch_rsi_reversal,
    "doom_volatility_contraction": doom_volatility_contraction,
    "volatility_ladder_strategy": volatility_ladder_strategy,
    "sentiment_volatility_stochastic_fade": sentiment_volatility_stochastic_fade
}


RULE_STRATEGY_NAMES = list(RULE_BASED_STRATEGIES.keys())
