# trading/confidence.py

import json
import os
from datetime import datetime

CONF_PATH = "data/strategy_confidence.json"
DEFAULT_CONFIDENCE = 0.5
DECAY = 0.5  # Controls how fast confidence moves (closer to 1 = slower)

_conf_data = {}


def _load_conf():
    global _conf_data
    if os.path.exists(CONF_PATH):
        with open(CONF_PATH, "r") as f:
            _conf_data = json.load(f)
    else:
        _conf_data = {}


def _save_conf():
    with open(CONF_PATH, "w") as f:
        json.dump(_conf_data, f, indent=2)


def get_confidence(strategy_name: str) -> float:
    _load_conf()
    entry = _conf_data.get(strategy_name, {})
    return entry.get("confidence", DEFAULT_CONFIDENCE)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def record_result(strategy_names: list, won: bool, decay: float = DECAY):
    """
    Update confidence for ALL strategies that participated in a consensus trade.
    Confidence moves faster when more strategies agree (consensus_factor > 1).
    """
    _load_conf()

    unique_strats = [s for s in set(strategy_names) if s]
    consensus_factor = 1.0 + 0.1 * max(0, len(unique_strats) - 1)
    result = 1.0 if won else 0.0

    for stra in unique_strats:
        old_entry = _conf_data.get(stra, {})
        old_conf = old_entry.get("confidence", DEFAULT_CONFIDENCE)

        # Move confidence toward result; amplify by consensus_factor, clamp to [0,1]
        delta = (result - old_conf) * (1 - decay) * consensus_factor
        new_conf = _clamp(old_conf + delta)

        _conf_data[stra] = {
            "confidence": new_conf,
            "last_updated": datetime.utcnow().isoformat()
        }

    _save_conf()


def reset_confidence(strategy_name: str = None):
    _load_conf()
    if strategy_name:
        _conf_data[strategy_name] = {
            "confidence": DEFAULT_CONFIDENCE,
            "last_updated": datetime.utcnow().isoformat()
        }
    else:
        for strat in _conf_data:
            _conf_data[strat] = {
                "confidence": DEFAULT_CONFIDENCE,
                "last_updated": datetime.utcnow().isoformat()
            }
    _save_conf()
